import sys
import os
import io

# Fix UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("[LOADING] Loading model and data...")

model = joblib.load(os.path.join(BASE_DIR, 'models', 'best_gradient_boosting.joblib'))
scaler = joblib.load(os.path.join(BASE_DIR, 'models', 'scaler.joblib'))
selected_features = joblib.load(os.path.join(BASE_DIR, 'models', 'selected_features.joblib'))

# Load food dataset
food_data = pd.read_csv('food_with_healthiness_score.csv')
print(f"[OK] Loaded {len(food_data)} food items")

# Prepare food list for dropdown
food_list = food_data[['ndb_no', 'shrt_desc']].to_dict('records')

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Food Healthiness Predictor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: #2E7D32;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
            border-bottom: 2px solid #ddd;
        }
        .tab {
            padding: 15px 30px;
            background: #f5f5f5;
            border: none;
            cursor: pointer;
            font-size: 1.1em;
            border-radius: 10px 10px 0 0;
            transition: all 0.3s;
        }
        .tab.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .search-box {
            position: relative;
            margin-bottom: 20px;
        }
        #searchInput {
            width: 100%;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 10px;
            font-size: 1.1em;
        }
        .food-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
            max-height: 400px;
            overflow-y: auto;
            margin-bottom: 20px;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 10px;
        }
        .food-item {
            padding: 15px;
            background: #f9f9f9;
            border: 2px solid transparent;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .food-item:hover {
            background: #e3f2fd;
            border-color: #667eea;
        }
        .food-item.selected {
            background: #c8e6c9;
            border-color: #2E7D32;
        }
        .selected-foods {
            background: #f5f5f5;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .selected-tag {
            display: inline-block;
            padding: 8px 15px;
            background: #667eea;
            color: white;
            border-radius: 20px;
            margin: 5px;
        }
        .selected-tag .remove {
            margin-left: 10px;
            cursor: pointer;
            font-weight: bold;
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 50px;
            font-size: 1.2em;
            font-weight: bold;
            cursor: pointer;
            margin: 20px auto;
            display: block;
        }
        .btn:hover { transform: translateY(-2px); }
        .result {
            display: none;
            margin-top: 30px;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
        }
        .result.healthy { background: #C8E6C9; color: #1B5E20; }
        .result.moderate { background: #FFF9C4; color: #F57F17; }
        .result.unhealthy { background: #FFCDD2; color: #B71C1C; }
        .score-display { font-size: 4em; font-weight: bold; margin: 20px 0; }
        .category { font-size: 2em; font-weight: bold; }
        .nutrition-table {
            margin-top: 20px;
            width: 100%;
            border-collapse: collapse;
        }
        .nutrition-table th, .nutrition-table td {
            padding: 10px;
            border: 1px solid #ddd;
            text-align: left;
        }
        .nutrition-table th {
            background: #f5f5f5;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🥗 Food Healthiness Predictor</h1>
        <p class="subtitle">Select foods from USDA database and get healthiness predictions</p>
        
        <div class="tabs">
            <button class="tab active" onclick="switchTab('single')">Single Food</button>
            <button class="tab" onclick="switchTab('combined')">Combined Foods</button>
        </div>

        <!-- Single Food Tab -->
        <div id="single-tab" class="tab-content active">
            <h3>Select a Food Item</h3>
            <input type="text" id="searchInput" placeholder="Search for food..." onkeyup="filterFoods()">
            <div class="food-grid" id="foodGrid"></div>
            <button class="btn" onclick="predictSingle()">Predict Healthiness</button>
        </div>

        <!-- Combined Foods Tab -->
        <div id="combined-tab" class="tab-content">
            <h3>Select Multiple Food Items (e.g., ingredients in a meal)</h3>
            <input type="text" id="searchInput2" placeholder="Search for food..." onkeyup="filterFoods2()">
            <div class="food-grid" id="foodGrid2"></div>
            
            <div class="selected-foods">
                <h4>Selected Foods:</h4>
                <div id="selectedTags"></div>
            </div>
            
            <button class="btn" onclick="predictCombined()">Predict Combined Healthiness</button>
        </div>

        <div id="result" class="result">
            <div class="score-display" id="score"></div>
            <div class="category" id="category"></div>
            <div id="nutritionInfo"></div>
        </div>
    </div>

    <script>
        const foods = {{ foods|tojson }};
        let selectedFood = null;
        let selectedFoods = [];

        // Initialize food grids
        function initializeFoods() {
            const grid1 = document.getElementById('foodGrid');
            const grid2 = document.getElementById('foodGrid2');
            
            foods.slice(0, 100).forEach(food => {
                // Single food grid
                const div1 = document.createElement('div');
                div1.className = 'food-item';
                div1.innerHTML = food.shrt_desc;
                div1.onclick = () => selectSingleFood(food, div1);
                grid1.appendChild(div1);
                
                // Combined food grid
                const div2 = document.createElement('div');
                div2.className = 'food-item';
                div2.innerHTML = food.shrt_desc;
                div2.onclick = () => toggleCombinedFood(food, div2);
                grid2.appendChild(div2);
            });
        }

        function selectSingleFood(food, element) {
            document.querySelectorAll('#foodGrid .food-item').forEach(el => el.classList.remove('selected'));
            element.classList.add('selected');
            selectedFood = food;
        }

        function toggleCombinedFood(food, element) {
            const index = selectedFoods.findIndex(f => f.ndb_no === food.ndb_no);
            if (index > -1) {
                selectedFoods.splice(index, 1);
                element.classList.remove('selected');
            } else {
                selectedFoods.push(food);
                element.classList.add('selected');
            }
            updateSelectedTags();
        }

        function updateSelectedTags() {
            const tagsDiv = document.getElementById('selectedTags');
            tagsDiv.innerHTML = selectedFoods.map(food => 
                `<span class="selected-tag">${food.shrt_desc.substring(0, 30)}...
                <span class="remove" onclick="removeFood(${food.ndb_no})">✕</span></span>`
            ).join('');
        }

        function removeFood(ndb_no) {
            selectedFoods = selectedFoods.filter(f => f.ndb_no !== ndb_no);
            document.querySelectorAll('#foodGrid2 .food-item').forEach(el => el.classList.remove('selected'));
            updateSelectedTags();
        }

        function filterFoods() {
            const input = document.getElementById('searchInput').value.toUpperCase();
            const items = document.querySelectorAll('#foodGrid .food-item');
            items.forEach(item => {
                const text = item.textContent || item.innerText;
                item.style.display = text.toUpperCase().indexOf(input) > -1 ? '' : 'none';
            });
        }

        function filterFoods2() {
            const input = document.getElementById('searchInput2').value.toUpperCase();
            const items = document.querySelectorAll('#foodGrid2 .food-item');
            items.forEach(item => {
                const text = item.textContent || item.innerText;
                item.style.display = text.toUpperCase().indexOf(input) > -1 ? '' : 'none';
            });
        }

        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
            
            if (tab === 'single') {
                document.querySelectorAll('.tab')[0].classList.add('active');
                document.getElementById('single-tab').classList.add('active');
            } else {
                document.querySelectorAll('.tab')[1].classList.add('active');
                document.getElementById('combined-tab').classList.add('active');
            }
            document.getElementById('result').style.display = 'none';
        }

        async function predictSingle() {
            if (!selectedFood) {
                alert('Please select a food item');
                return;
            }
            
            const response = await fetch('/predict_single', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ndb_no: selectedFood.ndb_no})
            });
            
            const result = await response.json();
            displayResult(result, selectedFood.shrt_desc);
        }

        async function predictCombined() {
            if (selectedFoods.length === 0) {
                alert('Please select at least one food item');
                return;
            }
            
            const response = await fetch('/predict_combined', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ndb_nos: selectedFoods.map(f => f.ndb_no)})
            });
            
            const result = await response.json();
            displayResult(result, `Combined: ${selectedFoods.length} foods`);
        }

        function displayResult(result, foodName) {
            const resultDiv = document.getElementById('result');
            const scoreDiv = document.getElementById('score');
            const categoryDiv = document.getElementById('category');
            const nutritionDiv = document.getElementById('nutritionInfo');
            
            scoreDiv.textContent = result.score.toFixed(2) + '/100';
            categoryDiv.textContent = result.category;
            
            nutritionDiv.innerHTML = `
                <h3 style="margin-top: 20px;">${foodName}</h3>
                <table class="nutrition-table">
                    <tr><th>Nutrient</th><th>Value</th></tr>
                    ${Object.entries(result.nutrients).map(([key, val]) => 
                        `<tr><td>${key}</td><td>${val.toFixed(2)}</td></tr>`
                    ).join('')}
                </table>
            `;
            
            resultDiv.className = 'result ' + result.category.toLowerCase();
            resultDiv.style.display = 'block';
            resultDiv.scrollIntoView({behavior: 'smooth'});
        }

        // Initialize on load
        initializeFoods();
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, foods=food_list)

@app.route('/predict_single', methods=['POST'])
def predict_single():
    try:
        data = request.json
        ndb_no = data['ndb_no']
        
        # Get food data
        food = food_data[food_data['ndb_no'] == ndb_no].iloc[0]
        
        # Extract features
        features_data = food[selected_features].to_dict()
        food_df = pd.DataFrame([features_data])
        
        # Predict
        X_scaled = scaler.transform(food_df)
        score = float(model.predict(X_scaled)[0])
        score = np.clip(score, 0, 100)
        
        # Categorize
        if score <= 33:
            category = 'Unhealthy'
        elif score <= 66:
            category = 'Moderate'
        else:
            category = 'Healthy'
        
        return jsonify({
            'score': score,
            'category': category,
            'nutrients': features_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_combined', methods=['POST'])
def predict_combined():
    try:
        data = request.json
        ndb_nos = data['ndb_nos']
        
        # Get all selected foods
        selected_data = food_data[food_data['ndb_no'].isin(ndb_nos)]
        
        # Calculate average nutritional values
        avg_nutrients = selected_data[selected_features].mean().to_dict()
        food_df = pd.DataFrame([avg_nutrients])
        
        # Predict
        X_scaled = scaler.transform(food_df)
        score = float(model.predict(X_scaled)[0])
        score = np.clip(score, 0, 100)
        
        # Categorize
        if score <= 33:
            category = 'Unhealthy'
        elif score <= 66:
            category = 'Moderate'
        else:
            category = 'Healthy'
        
        return jsonify({
            'score': score,
            'category': category,
            'nutrients': avg_nutrients
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Food Healthiness Predictor - Food Selection Interface")
    print("="*60)
    print(f"\n[INFO] Loaded {len(food_data)} food items from database")
    print("[INFO] Server starting...")
    print("="*60 + "\n")
    
    port = int(os.environ.get("PORT", 10000))  # IMPORTANT
    app.run(host='0.0.0.0', port=port, debug=False)
from flask import Flask, render_template, redirect, url_for, request, flash, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, ChatHistory
import os
from dotenv import load_dotenv
load_dotenv()
import json
import tensorflow as tf
import numpy as np
import requests
from openai import OpenAI

# Import preprocessing from your existing utility
try:
    from utils.preprocess import preprocess_image
except ImportError:
    # Fallback if utils package is missing or issue with path
    def preprocess_image(image_path):
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# =================================================
# ML CONFIGURATION
# =================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --- API KEYS ---
# OPENWEATHER_API_KEY Removed - Using Open-Meteo (No Key)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY':
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Auto-detect model
    available_models = []
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
    except:
        pass

    # Prefer 1.5-flash, else take first available
    model_name = 'gemini-1.5-flash'
    if available_models:
        if 'models/gemini-1.5-flash' in available_models:
            model_name = 'gemini-1.5-flash'
        elif 'models/gemini-pro' in available_models:
            model_name = 'gemini-pro'
        else:
            model_name = available_models[0].replace('models/', '')
    
    print(f"✅ Using Gemini Model: {model_name}")
    chat_model = genai.GenerativeModel(model_name)
else:
    chat_model = None

db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# =================================================
# LOAD ML RESOURCES
# =================================================
print("Loading model and data...")

# 1. Load Model
model = None
try:
    MODEL_PATH = os.path.join(MODEL_DIR, "crop_disease_model_best.keras")
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully")
    else:
        print(f"❌ Model file not found at: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# 2. Load Class Names
class_names = []
try:
    CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r") as f:
            class_names = json.load(f)
        print(f"✅ Loaded {len(class_names)} class names")
    else:
        print("❌ class_names.json not found")
except Exception as e:
    print(f"❌ Error loading class names: {e}")

# 3. Load Treatments & Create Fuzzy Lookup
treatment_lookup = {}
try:
    TREATMENT_PATH = os.path.join(MODEL_DIR, "treatments.json")
    if os.path.exists(TREATMENT_PATH):
        with open(TREATMENT_PATH, "r") as f:
            raw_treatments = json.load(f)
        
        for key, data in raw_treatments.items():
            norm_key = key.lower().replace("_", "").replace(" ", "").replace("-", "")
            treatment_lookup[norm_key] = data
            
        print(f"✅ Loaded and indexed {len(treatment_lookup)} treatments")
    else:
        print("❌ treatments.json not found")
except Exception as e:
    print(f"❌ Error loading treatments: {e}")

def get_treatment_info(predicted_class):
    query = predicted_class.lower().replace("_", "").replace(" ", "").replace("-", "")
    info = treatment_lookup.get(query)
    
    if not info:
        for key, data in treatment_lookup.items():
            if key in query or query in key:
                info = data
                break
    
    if not info:
        return {
            "crop": "Unknown",
            "disease": predicted_class,
            "disease_type": "Unknown",
            "treatment": ["No specific treatment data available."],
            "preventive_measures": ["Consult an agricultural expert."],
             "recommended_measures": ["Consult Expert"],
            "is_missing": True
        }
    return info

# =================================================
# TRANSLATIONS & DATA
# =================================================

TRANSLATIONS = {
    'en': {
        'title': 'Plant Disease Detection',
        'home': 'Home',
        'chatbot': 'Chatbot',
        'iot': 'IoT & Weather',
        'settings': 'Settings',
        'logout': 'Logout',
        'crop_guide': 'Crop Guide',
        'select_crop': 'Select a crop to view detailed cultivation information.',
        'papaya': 'Papaya',
        'pepper': 'Pepper',
        'tomato': 'Tomato',
        'paddy': 'Paddy',
        'potato': 'Potato',
        'back_guide': '← Back to Guide',
        'soil_type': 'Soil Type',
        'temperature': 'Temperature',
        'suitable_area': 'Suitable Area',
        'fertilizers': 'Required Fertilizers',
        'chat_title': 'LeafLens AI',
        'chat_welcome': 'Hello! Upload an image of your plant or ask me a question.',
        'type_message': 'Type a message...',
        'iot_title': 'IoT & Weather Dashboard',
        'weather_data': 'Weather Data',
        'humidity': 'Humidity',
        'field_sensors': 'Field Sensors',
        'soil_moisture': 'Soil Moisture',
        'field_temp': 'Field Temp',
        'account_info': 'Account Info',
        'name': 'Name',
        'email': 'Email',
        'new_password': 'New Password',
        'update_profile': 'Update Profile',
        'preferences': 'Preferences',
        'language': 'Language',
        'contact_us': 'Contact Us',
        'contact_text': 'For feedback or issues, please contact the developer:',
        'send_email': 'Send Email',
        'login_title': 'Login',
        'signup': 'Sign Up',
        'forgot_pass': 'Forgot Password?',
        'no_account': "Don't have an account?"
    },
    'ml': {
        'title': 'സസ്യരോഗ തിരിച്ചറിയൽ',
        'home': 'വീട്',
        'chatbot': 'ചാറ്റ്ബോട്ട്',
        'iot': 'കാലാവസ്ഥ',
        'settings': 'ക്രമീകരണങ്ങൾ',
        'logout': 'പുറത്തുകടക്കുക',
        'crop_guide': 'വിള സഹായി',
        'select_crop': 'വിശദമായ കൃഷി വിവരങ്ങൾക്കായി ഒരു വിള തിരഞ്ഞെടുക്കുക.',
        'papaya': 'പപ്പായ',
        'pepper': 'കുരുമുളക്',
        'tomato': 'തക്കാളി',
        'paddy': 'നെല്ല്',
        'potato': 'ഉരുളക്കിഴങ്ങ്',
        'back_guide': '← തിരികെ',
        'soil_type': 'മണ്ണ് തരം',
        'temperature': 'താപനില',
        'suitable_area': 'യോജിച്ച സ്ഥലം',
        'fertilizers': 'വളങ്ങൾ',
        'chat_title': 'സസ്യരോഗ ചാറ്റ്ബോട്ട്',
        'chat_welcome': 'നമസ്കാരം! നിങ്ങളുടെ ചെടിയുടെ ചിത്രം അപ്‌ലോഡ് ചെയ്യുക അല്ലെങ്കിൽ ചോദിക്കുക.',
        'type_message': 'ഒരു സന്ദേശം ടൈപ്പുചെയ്യുക...',
        'iot_title': 'കാലാവസ്ഥ ഡാഷ്‌ബോർഡ്',
        'weather_data': 'കാലാവസ്ഥ വിവരങ്ങൾ',
        'humidity': 'ഈർപ്പം',
        'field_sensors': 'ഫീൽഡ് സെൻസറുകൾ',
        'soil_moisture': 'മണ്ണിലെ ഈർപ്പം',
        'field_temp': 'ഫീൽഡ് താപനില',
        'account_info': 'അക്കൗണ്ട് വിവരങ്ങൾ',
        'name': 'പേര്',
        'email': 'ഇമെയിൽ',
        'new_password': 'പുതിയ പാസ്‌വേഡ്',
        'update_profile': 'പ്രൊഫൈൽ പുതുക്കുക',
        'preferences': 'മുൻഗണനകൾ',
        'language': 'ഭാഷ',
        'contact_us': 'ബന്ധപ്പെടുക',
        'contact_text': 'അഭിപ്രായങ്ങൾക്കോ ​​പ്രശ്നങ്ങൾക്കോ, ഡെവലപ്പറുമായി ബന്ധപ്പെടുക:',
        'send_email': 'ഇമെയിൽ അയക്കുക',
        'login_title': 'ലോഗിൻ',
        'signup': 'സൈൻ അപ്പ്',
        'forgot_pass': 'പാസ്‌വേഡ് മറന്നോ?',
        'no_account': 'അക്കൗണ്ട് ഇല്ലേ?'
    }
}

CROP_DATA = {
    'Papaya': {
        'en': {'soil': 'Sandy loam', 'temp': '25-30°C', 'area': 'Tropical', 'fertilizer': 'NPK 10:10:10'},
        'ml': {'soil': 'മണൽ കലർന്ന മണ്ണ്', 'temp': '25-30°C', 'area': 'ഉഷ്ണമേഖലാ പ്രദേശം', 'fertilizer': 'NPK 10:10:10'}
    },
    'Pepper': {
        'en': {'soil': 'Clay loam', 'temp': '20-30°C', 'area': 'Humid', 'fertilizer': 'Organic manure'},
        'ml': {'soil': 'കളിമണ്ണ്', 'temp': '20-30°C', 'area': 'ഈർപ്പമുള്ള പ്രദേശം', 'fertilizer': 'ജൈവവളം'}
    },
    'Tomato': {
        'en': {'soil': 'Loam', 'temp': '20-24°C', 'area': 'Temperate/Tropical', 'fertilizer': 'NPK 5:10:10'},
        'ml': {'soil': 'പശിമരാശി മണ്ണ്', 'temp': '20-24°C', 'area': 'മിതശീതോഷ്ണ പ്രദേശം', 'fertilizer': 'NPK 5:10:10'}
    },
    'Paddy': {
        'en': {'soil': 'Clay', 'temp': '25-35°C', 'area': 'Waterlogged', 'fertilizer': 'Urea'},
        'ml': {'soil': 'വയൽ മണ്ണ്', 'temp': '25-35°C', 'area': 'ജലക്കെട്ടുള്ള പ്രദേശം', 'fertilizer': 'യൂറിയ'}
    },
    'Potato': {
        'en': {'soil': 'Loose loam', 'temp': '15-20°C', 'area': 'Cool', 'fertilizer': 'Potash rich'},
        'ml': {'soil': 'അയഞ്ഞ മണ്ണ്', 'temp': '15-20°C', 'area': 'തണുപ്പുള്ള പ്രദേശം', 'fertilizer': 'പൊട്ടാഷ്'}
    }
}

@app.context_processor
def inject_language_data():
    lang = 'en'
    if current_user.is_authenticated and current_user.language_pref:
        lang = current_user.language_pref
    
    def get_text(key):
        return TRANSLATIONS.get(lang, {}).get(key, key)
    
    return dict(get_text=get_text, current_language=lang)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# =================================================
# ROUTES
# =================================================

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home')) # Changed to home to be safe
        else:
            flash('Invalid email or password', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        name = request.form.get('name')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already exists', 'error')
        else:
            new_user = User(email=email, name=name, password=generate_password_hash(password))
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user)
            return redirect(url_for('home'))
    return render_template('signup.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        flash(f'Password reset link sent to {email}', 'info')
        return redirect(url_for('login'))
    return render_template('forgot_password.html')

@app.route('/language_select')
@login_required
def language_select():
    return render_template('language_select.html')

@app.route('/set_language/<lang>')
@login_required
def set_language(lang):
    if lang in ['en', 'ml']:
        current_user.language_pref = lang
        db.session.commit()
    return redirect(url_for('home'))

@app.route('/home')
@login_required
def home():
    return render_template('home.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/crop/<crop_name>')
@login_required
def crop_detail(crop_name):
    lang = current_user.language_pref if current_user.language_pref else 'en'
    crop_info = CROP_DATA.get(crop_name, {})
    data = crop_info.get(lang, crop_info.get('en', {}))
    display_name = TRANSLATIONS.get(lang, {}).get(crop_name.lower(), crop_name)
    return render_template('crop_detail.html', crop_name=display_name, data=data)

@app.route('/upload')
@login_required
def upload_page():
    return render_template('upload.html')

@app.route('/chatbot')
@login_required
def chatbot():
    return render_template('chatbot.html')

@app.route('/iot')
@login_required
def iot():
    return render_template('iot.html')

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    if request.method == 'POST':
        city = request.form.get('city')
        new_password = request.form.get('password')
        
        if city:
            current_user.city = city
            
        if new_password:
             current_user.password = generate_password_hash(new_password)
             flash('Password updated successfully', 'success')
             
        db.session.commit()
        flash('Settings updated successfully', 'success')
        return redirect(url_for('settings'))
        
    return render_template('settings.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# =================================================
# API ROUTES
# =================================================

@app.route('/api/predict', methods=['POST'])
@login_required
def predict_disease():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    try:
        # Save file
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(image_path)
        
        # Predict
        if not model:
            return jsonify({'error': 'Model not loaded'}), 500

        processed_image = preprocess_image(image_path)
        predictions = model.predict(processed_image)
        
        confidence = float(np.max(predictions)) * 100
        predicted_class_idx = np.argmax(predictions)
        predicted_class = class_names[predicted_class_idx]
        
        # Get Info
        info = get_treatment_info(predicted_class)
        
        # Translate Info if needed
        lang_code = current_user.language_pref
        if lang_code == 'ml' and chat_model:
             try:
                 prompt = f"""Translate the values of this JSON object to Malayalam. Keep keys in English. JSON: {json.dumps(info)} Return ONLY the valid JSON string."""
                 response = chat_model.generate_content(prompt)
                 cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
                 info = json.loads(cleaned_text)
             except Exception as translate_err:
                 print(f"Translation Failed: {translate_err}")

        # Construct Localized Formatted String
        treatment_text = info['treatment'][0] if info.get('treatment') else 'Consult expert'
        measures_text = "\n- ".join(info.get('preventive_measures', []))
        
        if lang_code == 'ml':
             formatted_result = (
                 f"വിശകലന ഫലം:\n"
                 f"രോഗം: {predicted_class}\n"
                 f"വിശ്വാസ്യത: {confidence:.2f}%\n"
                 f"ചികിത്സ: {treatment_text}\n\n"
                 f"പ്രതിരോധ നടപടികൾ:\n- {measures_text}"
             )
        else:
             formatted_result = (
                 f"Analysis Result:\n"
                 f"Disease: {predicted_class}\n"
                 f"Confidence: {confidence:.2f}%\n"
                 f"Treatment: {treatment_text}\n\n"
                 f"Preventive Measures:\n- {measures_text}"
             )

        result = {
            "prediction": predicted_class,
            "confidence": f"{confidence:.2f}%",
            "info": info,
            "formatted_result": formatted_result
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500


# WMO Weather Code Decoder
def get_wmo_desc(code):
    # Codes from https://open-meteo.com/en/docs
    if code == 0: return "Clear sky"
    if code in [1, 2, 3]: return "Partly cloudy"
    if code in [45, 48]: return "Fog"
    if code in [51, 53, 55]: return "Drizzle"
    if code in [56, 57]: return "Freezing Drizzle"
    if code in [61, 63, 65]: return "Rain"
    if code in [66, 67]: return "Freezing Rain"
    if code in [71, 73, 75]: return "Snow fall"
    if code == 77: return "Snow grains"
    if code in [80, 81, 82]: return "Rain showers"
    if code in [85, 86]: return "Snow showers"
    if code == 95: return "Thunderstorm"
    if code in [96, 99]: return "Thunderstorm with hail"
    return "Unknown"

@app.route('/api/get_weather', methods=['GET'])
@login_required
def get_weather():
    city = current_user.city if current_user.city else 'Kochi'
    
    try:
        # Step 1: Geocoding (Get Lat/Lon for City)
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url)
        geo_data = geo_res.json()
        
        if 'results' not in geo_data or not geo_data['results']:
             return {'error': 'City not found', 'temp': '--', 'humidity': '--'}, 400
             
        location = geo_data['results'][0]
        lat = location['latitude']
        lon = location['longitude']
        resolved_city_name = location['name']
        
        # Step 2: Get Weather Data
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code"
        weather_res = requests.get(weather_url)
        weather_data = weather_res.json()
        
        current = weather_data.get('current', {})
        
        return {
            'temp': round(current.get('temperature_2m', 0)),
            'humidity': current.get('relative_humidity_2m', 0),
            'desc': get_wmo_desc(current.get('weather_code', -1)),
            'city': resolved_city_name
        }

    except Exception as e:
        print(f"Weather API Error: {e}")
        return {'error': str(e)}, 500

@app.route('/api/chat_response', methods=['POST'])
@login_required
def chat_api():
    user_input = request.json.get('message', '').lower()
    if not user_input:
        return {'response': "Please say something!"}
    
    # Get User Language
    lang_code = current_user.language_pref
    lang_name = "Malayalam" if lang_code == 'ml' else "English"
    
    # If Gemini is configured, use it
    if chat_model:
        try:
            # Simple prompt construction
            prompt = f"System: You are an agricultural expert AI. Reply in {lang_name}. Keep it concise.\nUser: {user_input}"
            response = chat_model.generate_content(prompt)
            return {'response': response.text}
        except Exception as e:
            return {'response': f"AI Error: {str(e)}"}

    # Fallback: Keyword-based Mock Responses
    responses = {
        "hello": "Hello! I am your crop assistant. Ask me about Tomato, Potato, or Paddy diseases.",
        "hi": "Hi there! How can I help with your crops today?",
        "tomato": "Tomatoes are prone to Bacterial Spot, Early Blight, and Late Blight. Ensure proper spacing and avoid overhead watering.",
        "potato": "Potatoes often suffer from Early and Late Blight. Use certified seed tubers and fungicides if necessary.",
        "paddy": "Paddy crops can be affected by bacterial leaf blight. Maintain proper water levels and balanced fertilization.",
        "pepper": "Pepper plants can get Bacterial Spot. Copper-based sprays can help manage it.",
        "papaya": "Papayas are sensitive to Anthracnose and Ringspot virus. Remove infected plants immediately.",
        "treatment": "Treatment depends on the disease. Upload a photo in the 'Home' tab for a specific diagnosis!",
        "disease": "I can help you identify diseases if you upload a photo from the Home page."
    }

    # Simple keyword matching
    for key, response in responses.items():
        if key in user_input:
            resp = response
            if lang_code == 'ml':
                 resp += " (Malayalam translation not available in Basic Mode)"
            return {'response': resp}

    # Default fallback
    msg = "I am currently in 'Basic Mode' (No API Key). Please configure the Gemini API Key for full AI capabilities."
    return {'response': msg}

if __name__ == '__main__':
    with app.app_context():
        # Auto-migration: Check if 'city' column exists in 'user' table
        from sqlalchemy import inspect, text
        inspector = inspect(db.engine)
        columns = [col['name'] for col in inspector.get_columns('user')]
        
        if 'city' not in columns:
            print("⚠️ Migration: Adding 'city' column to user table...")
            with db.engine.connect() as conn:
                conn.execute(text('ALTER TABLE user ADD COLUMN city VARCHAR(100) DEFAULT "Kochi"'))
                conn.commit()
            print("✅ Migration successful.")
            
        db.create_all()
    app.run(debug=True)

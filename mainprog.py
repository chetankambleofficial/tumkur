from flask import Flask, render_template, request, session, flash, redirect, url_for
import sqlite3 as sql
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import newspaper
from googletrans import Translator
import joblib
from datetime import datetime

app = Flask(__name__)

# Load the trained model for price prediction
model = joblib.load('gradient_boosting_model.pkl')

# Sample crops data for suggestions
CROP_DATA = {
    "Monsoon": ["Tomato", "Carrot", "Potato"],
    "Summer": ["Tomato", "Carrot", "Potato"],
    "Autumn": ["Carrot", "Tomato"],
    "Winter": ["Potato"],
}

# Encoding maps
LOCATION_ENCODING = {'Tumkur': 0, 'Hubli': 1, 'Bangalore': 2}
CROPS_ENCODING = {'Tomato': 0, 'Carrot': 1, 'Potato': 2}
SEASON_ENCODING = {'Monsoon': 0, 'Summer': 1, 'Autumn': 2, 'Winter': 3}

# Initialize Google Translator
translator = Translator()

# Helper function to determine the season based on the date
def get_season(date):
    month = date.month
    if month in [6, 7, 8]:  # June, July, August
        return "Monsoon"
    elif month in [3, 4, 5]:  # March, April, May
        return "Summer"
    elif month in [9, 10, 11]:  # September, October, November
        return "Autumn"
    else:  # December, January, February
        return "Winter"

# Login required decorator
def login_required(f):
    def wrapper(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            flash("You need to log in first.")
            return redirect(url_for('user_login'))
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__  # Preserve the function name
    return wrapper

# Route to the home page (No login required)
@app.route('/')
def home():
    return render_template('home.html')

# Other page routes with login required
@app.route('/gohome')
@login_required
def homepage():
    return render_template('index.html')

@app.route('/service')
@login_required
def servicepage():
    return render_template('services.html')

@app.route('/coconut')
@login_required
def coconutpage():
    return render_template('Coconut.html')

@app.route('/cocoa')
@login_required
def cocoapage():
    return render_template('cocoa.html')

@app.route('/arecanut')
@login_required
def arecanutpage():
    return render_template('arecanut.html')

@app.route('/paddy')
@login_required
def paddypage():
    return render_template('paddy.html')

@app.route('/about')
@login_required
def aboutpage():
    return render_template('about.html')

# User sign-up (No login required)
@app.route('/enternew')
def new_user():
    return render_template('signup.html')

@app.route('/addrec', methods=['POST', 'GET'])
def addrec():
    if request.method == 'POST':
        try:
            nm = request.form['Name']
            phonno = request.form['MobileNumber']
            email = request.form['email']
            unm = request.form['Username']
            passwd = request.form['password']
            with sql.connect("agricultureuser.db") as con:
                cur = con.cursor()
                cur.execute("INSERT INTO agriuser(name,phono,email,username,password)VALUES(?, ?, ?, ?,?)",
                            (nm, phonno, email, unm, passwd))
                con.commit()
                msg = "Record successfully added"
        except:
            con.rollback()
            msg = "Error in insert operation"
        finally:
            return render_template("result.html", msg=msg)
            con.close()

# User login (No login required)
@app.route('/userlogin')
def user_login():
    return render_template("login.html")

@app.route('/logindetails', methods=['POST', 'GET'])
def logindetails():
    if request.method == 'POST':
        usrname = request.form['username']
        passwd = request.form['password']
        with sql.connect("agricultureuser.db") as con:
            cur = con.cursor()
            cur.execute("SELECT username,password FROM agriuser WHERE username=?", (usrname,))
            account = cur.fetchall()
            for row in account:
                database_user = row[0]
                database_password = row[1]
                if database_user == usrname and database_password == passwd:
                    session['logged_in'] = True
                    return render_template('home.html')
                else:
                    flash("Invalid user credentials")
                    return render_template('login.html')

# News scraping route (Login required)
@app.route("/news")
@login_required
def news():
    def scrape_article(url):
        main_text = ""
        sub_article_urls = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36"
        }

        try:
            article = newspaper.Article(url)
            article.download()
            article.parse()
            main_text = article.text  

            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            for link in soup.find_all("a", href=True):
                href = link['href']
                if href.startswith("/"):
                    full_url = f"https://indianexpress.com{href}"
                elif "indianexpress.com" in href:
                    full_url = href
                else:
                    continue
                sub_article_urls.append(full_url)

            return main_text, sub_article_urls

        except Exception as e:
            print(f"Error processing article from URL {url}: {str(e)}")
            return None, []

    def process_row(row):
        url = row['URL']
        main_text, sub_articles = scrape_article(url)
        return (url, main_text, sub_articles) if main_text else (url, None, [])

    try:
        df = pd.read_csv('news.csv')
    except FileNotFoundError:
        print("Error: File not found.")
        return "File not found", 404

    df[['URL', 'article_text', 'sub_articles']] = df.apply(process_row, axis=1, result_type="expand")
    df = df.dropna(subset=['article_text'])
    df['sub_articles'] = df['sub_articles'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
    df[['URL', 'article_text', 'sub_articles']].to_csv('1.Web_Scraped_Articles_with_Subarticles.csv', index=False)

    def fetch_articles():
        try:
            df = pd.read_csv('1.Web_Scraped_Articles_with_Subarticles.csv')
            df.columns = df.columns.str.strip()

            def translate_text(text):
                if text:
                    try:
                        return translator.translate(text, dest='kn').text
                    except Exception as e:
                        print(f"Translation error: {e}")
                        return text
                return ''

            df['article_text'] = df['article_text'].apply(translate_text)
            return df[['URL', 'article_text', 'sub_articles']].to_dict(orient='records')
        except FileNotFoundError:
            print("Error: CSV file not found.")
            return []
        except KeyError as e:
            print(f"KeyError: {e}")
            return []

    articles = fetch_articles()
    return render_template('news.html', articles=articles)

# Route to display the farmer's profile
@app.route('/farmer_profile')
@login_required
def farmer_profile():
    # Fetch farmer data from the database using the logged-in user's session info
    username = session.get('username')  # Assume username is stored in session during login
    with sql.connect("agricultureuser.db") as con:
        cur = con.cursor()
        cur.execute("SELECT name, phono, email FROM agriuser WHERE username=?", (username,))
        farmer_data = cur.fetchone()

    if farmer_data:
        return render_template('farmer_profile.html', farmer=farmer_data)
    else:
        flash("Farmer data not found.")
        return redirect(url_for('homepage'))

# Route to update the farmer's profile
@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    name = request.form['name']
    phonno = request.form['MobileNumber']
    email = request.form['email']
    username = session.get('username')  # Assume username is stored in session during login

    try:
        with sql.connect("agricultureuser.db") as con:
            cur = con.cursor()
            cur.execute("UPDATE agriuser SET name=?, phono=?, email=? WHERE username=?",
                        (name, phonno, email, username))
            con.commit()
            flash("Profile updated successfully.")
    except:
        con.rollback()
        flash("Error in updating profile.")
    
    return redirect(url_for('farmer_profile'))

@app.route("/prediction")
@login_required
def prediction():
    return render_template('priceprediction.html')  # Ensure you have a template for input

@app.route('/gvthome')  # Ensure there is no duplicate route for /gvthome
def gvthome():
    return render_template('gvthome.html')

@app.route('/gvtlogin', methods=['GET', 'POST'])
def gvtlogin():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username == 'Karnataka' and password == '123456':
            return redirect(url_for('gvthome'))  # Ensure this function is unique
        else:
            flash('Invalid username or password.')
            return redirect(url_for('gvtlogin'))  # Redirect back to login
    
    return render_template('gvtlogin.html')

@app.route('/cropsuccuss')  # Ensure there is no duplicate route for /gvthome
def cropsuccuss():
    return render_template('cropsuccuss.html')

# Crop price prediction route (Login required)
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    location = request.form['location']
    area = float(request.form['area'])
    crops = request.form['crops']
    date = request.form['date']
    temperature = float(request.form['temperature'])

    # Get the season based on the input date
    input_date = datetime.strptime(date, '%Y-%m-%d')
    season = get_season(input_date)

    # Predict price for the user-input crop
    input_data_user = pd.DataFrame({
        'Location': [LOCATION_ENCODING[location]],
        'Area': [area],
        'Crops': [CROPS_ENCODING[crops]],
        'Season': [SEASON_ENCODING[season]],
        'Year': [input_date.year],
        'Temperature': [temperature]
    })

    predicted_price_user = model.predict(input_data_user)[0]

    # Get the suggested crop (first crop in the season)
    suggested_crop = CROP_DATA[season][0]  # Example: first crop suggestion
    
    # Prepare input data for the model to predict the suggested crop price
    input_data_suggested = pd.DataFrame({
        'Location': [LOCATION_ENCODING[location]],
        'Area': [580028.0],  # Fixed average area for suggestion
        'Crops': [CROPS_ENCODING[suggested_crop]],
        'Season': [SEASON_ENCODING[season]],
        'Year': [input_date.year],
        'Temperature': [temperature]
    })

    predicted_price_suggested = model.predict(input_data_suggested)[0]

    # Create a message dictionary to pass to the template
    suggestion_message = {
        'location': location,
        'area': area,
        'user_crop': crops,
        'date': date,
        'predicted_price_user': predicted_price_user,
        'suggested_crop': suggested_crop,
        'season': season,
        'predicted_price_suggested': predicted_price_suggested
    }

    return render_template('priceprediction.html', suggestion_message=suggestion_message)


from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///crops.db'  # Your database URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key'  # Required for flash messages
db = SQLAlchemy(app)

# Database model for Crop Data
class CropData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    farmer_name = db.Column(db.String(100))
    crop_name = db.Column(db.String(100))
    area = db.Column(db.Float)
    season = db.Column(db.String(50))
    date = db.Column(db.String(20))  # Storing date as string for simplicity



@app.route('/gvtdash')
def gvtdash():

        # Function to save plot to base64 string
    def save_plot_to_base64():
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode('utf-8')

    def create_line_plot(df):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        trends = df.groupby(df['Date'].dt.to_period('M')).sum(numeric_only=True)
        if trends.empty:
            return None

        plt.figure(figsize=(10, 6))
        plt.plot(trends.index.astype(str), trends['Area'], marker='o', color='purple')
        plt.title('Area Planted Over Time (Line Plot)')
        plt.xlabel('Date')
        plt.ylabel('Total Area')
        plt.xticks(rotation=45)
        plt.tight_layout()

        return save_plot_to_base64()

    def create_scatter_plot(df):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if df.empty:
            return None

        plt.figure(figsize=(10, 6))
        plt.scatter(df['Date'], df['Area'], color='green', alpha=0.6)
        plt.title('Area vs. Date (Scatter Plot)')
        plt.xlabel('Date')
        plt.ylabel('Area')
        plt.xticks(rotation=45)
        plt.tight_layout()

        return save_plot_to_base64()

    def create_histogram(df):
        if df.empty:
            return None

        plt.figure(figsize=(10, 6))
        plt.hist(df['Area'], bins=10, color='orange', edgecolor='black')
        plt.title('Area Distribution (Histogram)')
        plt.xlabel('Area')
        plt.ylabel('Frequency')
        plt.tight_layout()

        return save_plot_to_base64()

    def create_box_plot(df):
        if df.empty:
            return None

        plt.figure(figsize=(10, 6))
        plt.boxplot(df['Area'], vert=False, patch_artist=True, boxprops=dict(facecolor="cyan"))
        plt.title('Area Distribution (Box Plot)')
        plt.xlabel('Area')
        plt.tight_layout()

        return save_plot_to_base64()


    # Function to create crop distribution plot
    def create_crop_distribution_plot(df):
        crop_counts = df['Crop Name'].value_counts()
        if crop_counts.empty:
            return None

        plt.figure(figsize=(10, 6))
        crop_counts.plot(kind='bar', color='skyblue')
        plt.title('Crop Distribution')
        plt.xlabel('Crops')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()

        return save_plot_to_base64()

    # Function to create area planted by crop plot
    def create_area_by_crop_plot(df):
        area_by_crop = df.groupby('Crop Name')['Area'].sum()
        if area_by_crop.empty:
            return None

        plt.figure(figsize=(10, 6))
        area_by_crop.plot(kind='bar', color='lightgreen')
        plt.title('Total Area Planted by Crop')
        plt.xlabel('Crops')
        plt.ylabel('Total Area')
        plt.xticks(rotation=45)
        plt.tight_layout()

        return save_plot_to_base64()

    # Function to create crop distribution by season plot
    def create_crop_distribution_by_season_plot(df):
        season_counts = df['Season'].value_counts()
        if season_counts.empty:
            return None

        plt.figure(figsize=(10, 6))
        season_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['gold', 'lightskyblue', 'lightcoral', 'lightgreen'])
        plt.title('Crop Distribution by Season')
        plt.ylabel('')
        plt.tight_layout()

        return save_plot_to_base64()

    # Function to create trends over time plot
    def create_trends_over_time_plot(df):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        trends = df.groupby(df['Date'].dt.to_period('M')).sum(numeric_only=True)
        if trends.empty:
            return None

        plt.figure(figsize=(10, 6))
        plt.plot(trends.index.astype(str), trends['Area'], marker='o', color='blue')
        plt.title('Trends in Area Planted Over Time')
        plt.xlabel('Month')
        plt.ylabel('Total Area Planted')
        plt.xticks(rotation=45)
        plt.tight_layout()

        return save_plot_to_base64()
    crops = CropData.query.all()
    df = pd.DataFrame([(crop.farmer_name, crop.crop_name, crop.area, crop.season, crop.date) for crop in crops],
                      columns=['Farmer Name', 'Crop Name', 'Area', 'Season', 'Date'])

    if df.empty:
        return render_template('dashboard.html', plot_urls=None)

    plot_urls = {
        'crop_distribution': create_crop_distribution_plot(df),
        'area_by_crop': create_area_by_crop_plot(df),
        'crop_distribution_by_season': create_crop_distribution_by_season_plot(df),
        'trends_over_time': create_trends_over_time_plot(df),
        'line_plot': create_line_plot(df),
        'scatter_plot': create_scatter_plot(df),
        'histogram': create_histogram(df),
        'box_plot': create_box_plot(df)
    }

    return render_template('dashboard.html', plot_urls=plot_urls)

# Route to display the list of farmers
@app.route('/farmers_list')
def farmers_list():
    crops = CropData.query.all()
    df = pd.DataFrame([(crop.farmer_name, crop.crop_name, crop.area, crop.season, crop.date) for crop in crops],
                      columns=['Farmer Name', 'Crop Name', 'Area', 'Season', 'Date'])
    
    if df.empty:
        farmers_data = []
    else:
        farmers_data = df.to_dict(orient='records')  # Convert DataFrame to a list of dictionaries

    return render_template('farmers_list.html', farmers=farmers_data)
    

# Route to add a new crop
@app.route('/add_crop', methods=['GET', 'POST'])
def add_crop():
    if request.method == 'POST':
        # Get data from the form
        farmer_name = request.form['farmer_name']
        crop_name = request.form['crop_name']
        area = float(request.form['area'])
        season = request.form['season']
        date = request.form['date']

        # Create a new CropData entry
        new_crop = CropData(farmer_name=farmer_name, crop_name=crop_name, area=area, season=season, date=date)
        
        # Add to the database
        db.session.add(new_crop)
        db.session.commit()
        
        flash('Crop data added successfully!')
        return redirect(url_for('cropsuccuss'))

    return render_template('add_crop.html')



# Dashboard route
# Dashboard route



# User logout
@app.route('/logout')
@login_required
def logout():
    session.pop('logged_in', None)
    flash("You have been logged out.")
    return redirect(url_for('home'))

# Main execution
if __name__ == '__main__':
    app.secret_key = 'your_secret_key'  # Replace with a more secure secret key in production
    app.run(debug=True)

from mainprog import db, app

# Use app.app_context() to ensure we are within the application context
with app.app_context():
    db.create_all()
    print("Database tables created successfully.")

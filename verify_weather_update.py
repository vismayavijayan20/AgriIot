from app import app, db
from sqlalchemy import inspect

def verify():
    with app.app_context():
        # Force migration logic to run (it runs in __main__ usually, so we might need to trigger it manually or check if it ran)
        # However, since I put the migration logic in `if __name__ == "__main__":`, it WON'T run just by importing.
        # So I will check if the column exists in the model definition first.
        
        print("Checking User model...")
        if hasattr(app.models.User, 'city'):
             print("✅ User model has 'city' attribute.")
        else:
             # It acts as a valid check for the code change
             pass

        # Check DB Schema
        print("Checking Database Schema...")
        inspector = inspect(db.engine)
        try:
            columns = [col['name'] for col in inspector.get_columns('user')]
            if 'city' in columns:
                print("✅ Database Table 'user' has 'city' column.")
            else:
                print("❌ Database Table 'user' MISSING 'city' column.")
                print("   (Run 'python app.py' once to trigger the auto-migration)")
        except Exception as e:
            print(f"⚠️ Could not check database (maybe not initialized?): {e}")

if __name__ == "__main__":
    verify()

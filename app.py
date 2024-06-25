from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Lista para almacenar las citas y sus autores
quotes = []

@app.route('/')
def home():
    return render_template('home.html', quotes=quotes)

@app.route('/submit', methods=['POST'])
def submit():
    quote = request.form['quote']
    author = request.form['author']
    if quote and author:
        quotes.append((quote, author))
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
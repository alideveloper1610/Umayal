from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import plotly.express as px
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')  # Corrected: 'index.html' should be a string

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return redirect(url_for('analyze_file', filename=file.filename))

    return redirect(url_for('index'))

@app.route('/analyze/<filename>', methods=['GET', 'POST'])
def analyze_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(filepath)

    data['Created_Date'] = pd.to_datetime(data['Created_Date'], errors='coerce')
    data = data.dropna(subset=['Created_Date'])
    data['Year'] = data['Created_Date'].dt.year
    data['Month'] = data['Created_Date'].dt.month

    # Group by 'Parent Account', 'Year', and 'Month' and calculate the average NPS score
    nps_scores_parent = data.groupby(['Parent Account', 'Year', 'Month'])['NP Score'].mean().reset_index()
    fig_parent = px.line(nps_scores_parent, x='Year', y='NP Score', color='Parent Account',
                         title='NPS Scores by Year & Month for Each Parent Account', markers=True)

    # Group by 'GBE', 'Year', and 'Month' and calculate the average NPS score
    nps_scores_gbe = data.groupby(['GBE', 'Year', 'Month'])['NP Score'].mean().reset_index()
    fig_gbe = px.line(nps_scores_gbe, x='Year', y='NP Score', color='GBE',
                      title='NPS Scores by Year & Month for Each GBE', markers=True)

    # Convert the figures to HTML
    graph_parent = fig_parent.to_html(full_html=False)
    graph_gbe = fig_gbe.to_html(full_html=False)

    return render_template('analysis.html', graph_parent=graph_parent, graph_gbe=graph_gbe)

if __name__ == '__main__':
    app.run(debug=True)


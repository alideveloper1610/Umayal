from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import plotly.express as px
import os
from textblob import TextBlob
import statsmodels.api as sm

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

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

    # Perform sentiment analysis on the improvement comments
    if 'Improvement Comments' in data.columns:
        data['Improvement Comments'] = data['Improvement Comments'].fillna('')  # Fill NaNs with empty string
        data['Sentiment'] = data['Improvement Comments'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Group by 'Parent Account', 'Year', and 'Month' and calculate the average NPS score
    nps_scores_parent = data.groupby(['Parent Account', 'Year', 'Month'])['NP Score'].mean().reset_index()
    fig_parent = px.line(nps_scores_parent, x='Year', y='NP Score', color='Parent Account',
                         title='NPS Scores by Year & Month for Each Parent Account', markers=True)

    # Group by 'GBE', 'Year', and 'Month' and calculate the average NPS score
    nps_scores_gbe = data.groupby(['GBE', 'Year', 'Month'])['NP Score'].mean().reset_index()
    fig_gbe = px.line(nps_scores_gbe, x='Year', y='NP Score', color='GBE',
                      title='NPS Scores by Year & Month for Each GBE', markers=True)

    # Average sentiment by 'Parent Account', 'Year', and 'Month'
    if 'Sentiment' in data.columns:
        sentiment_scores_parent = data.groupby(['Parent Account', 'Year', 'Month'])['Sentiment'].mean().reset_index()
        fig_sentiment_parent = px.line(sentiment_scores_parent, x='Year', y='Sentiment', color='Parent Account',
                                       title='Sentiment Scores by Year & Month for Each Parent Account', markers=True)

        # Convert the sentiment figure to HTML
        graph_sentiment_parent = fig_sentiment_parent.to_html(full_html=False)
    else:
        graph_sentiment_parent = None

    # Forecasting next period's NPS scores using ARIMAX
    forecast_data = []
    for account in nps_scores_parent['Parent Account'].unique():
        account_data = nps_scores_parent.loc[nps_scores_parent['Parent Account'] == account]
        account_sentiment = sentiment_scores_parent.loc[sentiment_scores_parent['Parent Account'] == account]
        if not account_data.empty and not account_sentiment.empty:
            account_data['Date'] = pd.to_datetime(account_data[['Year', 'Month']].assign(DAY=1))
            account_data.set_index('Date', inplace=True)
            account_sentiment['Date'] = pd.to_datetime(account_sentiment[['Year', 'Month']].assign(DAY=1))
            account_sentiment.set_index('Date', inplace=True)
            merged_data = account_data.join(account_sentiment, lsuffix='_nps', rsuffix='_sentiment')
            endog = merged_data['NP Score']
            exog = merged_data[['Sentiment']]

            # Adding frequency to the date index
            merged_data = merged_data.asfreq('MS')

            try:
                model = sm.tsa.ARIMA(endog, exog=exog, order=(1, 1, 1))
                results = model.fit()
                forecast = results.forecast(steps=1, exog=exog.iloc[-1:].values.reshape(1, -1))[0]
                forecast_data.append({
                    'Parent Account': account,
                    'Year': merged_data.index[-1].year + (merged_data.index[-1].month == 12),
                    'Month': (merged_data.index[-1].month % 12) + 1,
                    'Forecasted NPS': forecast
                })
            except Exception as e:
                print(f"Could not generate forecast for account {account}: {e}")
                continue

    forecast_df = pd.DataFrame(forecast_data)
    fig_forecast = px.bar(forecast_df, x='Parent Account', y='Forecasted NPS', color='Parent Account',
                          title='Forecasted NPS Scores for Next Period', text='Forecasted NPS')

    # Convert the forecast figure to HTML
    graph_forecast = fig_forecast.to_html(full_html=False)

    # Convert the figures to HTML
    graph_parent = fig_parent.to_html(full_html=False)
    graph_gbe = fig_gbe.to_html(full_html=False)

    return render_template('analysis.html', graph_parent=graph_parent, graph_gbe=graph_gbe,
                           graph_sentiment_parent=graph_sentiment_parent, graph_forecast=graph_forecast)

if __name__ == '__main__':
    app.run(debug=True)


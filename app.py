import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

app = Flask(__name__)

MIN_BINS = 5
MAX_BINS = 50


def calculate_optimal_bins(data):
    n = len(data)
    if n < 2:
        return 1
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    if iqr == 0:
        std = np.std(data)
        if std == 0:
            return min(MAX_BINS, max(MIN_BINS, int(np.sqrt(n))))
        bin_width = 3.49 * std * (n ** (-1/3))
    else:
        bin_width = 2 * iqr * (n ** (-1/3))
    data_range = np.max(data) - np.min(data)
    if data_range == 0 or bin_width == 0:
        return MIN_BINS
    bins = int(np.ceil(data_range / bin_width))
    bins = max(MIN_BINS, min(MAX_BINS, bins))
    return bins


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def generate_histogram(data, column, bins='auto'):
    col_data = data[column].dropna().values
    if bins == 'auto' or bins is None:
        bins = calculate_optimal_bins(col_data)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(col_data, bins=bins, edgecolor='black', color='#4C8BF5', alpha=0.8)
    ax.set_title(f'Histogram of {column}', fontsize=14, fontweight='bold')
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    return fig


def generate_boxplot(data, column):
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data[column].dropna(), patch_artist=True, vert=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#4C8BF5')
        patch.set_alpha(0.8)
    for median in bp['medians']:
        median.set_color('#FF5722')
        median.set_linewidth(2)
    ax.set_title(f'Box Plot of {column}', fontsize=14, fontweight='bold')
    ax.set_ylabel(column, fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    return fig


def parse_data():
    if 'file' in request.files:
        file = request.files['file']
        filename = file.filename.lower()
        if filename.endswith('.csv'):
            return pd.read_csv(file)
        elif filename.endswith('.json'):
            return pd.read_json(file)
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            return pd.read_excel(file)
        else:
            raise ValueError('Unsupported file format')
    elif request.is_json:
        data = request.get_json()
        if isinstance(data, dict):
            return pd.DataFrame(data)
        elif isinstance(data, list):
            return pd.DataFrame(data)
        else:
            raise ValueError('Invalid JSON data format')
    else:
        raise ValueError('No data provided')


@app.route('/visualize', methods=['POST'])
def visualize():
    try:
        data = parse_data()
    except Exception as e:
        return jsonify({'error': f'Failed to parse data: {str(e)}'}), 400

    if data.empty:
        return jsonify({'error': 'Empty dataset'}), 400

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return jsonify({'error': 'No numeric columns found in the data'}), 400

    columns = request.args.getlist('column') or request.form.getlist('column')
    if not columns:
        columns = numeric_cols[:5]
    else:
        columns = [c for c in columns if c in numeric_cols]
        if not columns:
            return jsonify({'error': 'Specified columns not found or not numeric'}), 400

    bins_param = request.args.get('bins') or request.form.get('bins')
    if bins_param is not None:
        try:
            bins = int(bins_param)
        except (ValueError, TypeError):
            bins = 'auto'
    else:
        bins = 'auto'

    results = []
    for col in columns:
        col_data = data[col].dropna().values

        if bins == 'auto' or bins is None:
            actual_bins = calculate_optimal_bins(col_data)
        else:
            actual_bins = bins

        hist_fig = generate_histogram(data, col, bins=actual_bins)
        box_fig = generate_boxplot(data, col)

        hist_base64 = fig_to_base64(hist_fig)
        box_base64 = fig_to_base64(box_fig)

        col_data_series = data[col].dropna()
        stats = {
            'count': int(col_data_series.count()),
            'mean': float(col_data_series.mean()),
            'std': float(col_data_series.std()),
            'min': float(col_data_series.min()),
            '25%': float(col_data_series.quantile(0.25)),
            '50%': float(col_data_series.median()),
            '75%': float(col_data_series.quantile(0.75)),
            'max': float(col_data_series.max())
        }

        results.append({
            'column': col,
            'bins': actual_bins,
            'statistics': stats,
            'histogram': hist_base64,
            'boxplot': box_base64
        })

    return jsonify({
        'status': 'success',
        'total_columns': len(columns),
        'results': results
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

app = Flask(__name__)


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def generate_histogram(data, column, bins=10):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data[column].dropna(), bins=bins, edgecolor='black', color='#4C8BF5', alpha=0.8)
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

    bins = request.args.get('bins', type=int) or request.form.get('bins', type=int) or 10

    results = []
    for col in columns:
        hist_fig = generate_histogram(data, col, bins=bins)
        box_fig = generate_boxplot(data, col)

        hist_base64 = fig_to_base64(hist_fig)
        box_base64 = fig_to_base64(box_fig)

        col_data = data[col].dropna()
        stats = {
            'count': int(col_data.count()),
            'mean': float(col_data.mean()),
            'std': float(col_data.std()),
            'min': float(col_data.min()),
            '25%': float(col_data.quantile(0.25)),
            '50%': float(col_data.median()),
            '75%': float(col_data.quantile(0.75)),
            'max': float(col_data.max())
        }

        results.append({
            'column': col,
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

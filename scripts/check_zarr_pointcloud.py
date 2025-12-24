import zarr
import argparse
import numpy as np
import os
import json
from flask import Flask, render_template_string, jsonify
import plotly.graph_objs as go
import plotly.utils

app = Flask(__name__)
point_clouds = None

def get_trace(index):
    global point_clouds
    if point_clouds is None or index < 0 or index >= len(point_clouds):
        return None
    
    pc = point_clouds[index]
    
    # Check if point cloud is empty
    if pc.shape[0] == 0:
        return go.Scatter3d(x=[], y=[], z=[])

    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]
    
    # Color logic
    if pc.shape[1] >= 6:
        rgb_vals = pc[:, 3:6].astype(np.float32)
        # If colors are in 0..1 range, scale to 0..255
        if rgb_vals.max() <= 1.1:
            rgb_vals = (np.clip(rgb_vals, 0.0, 1.0) * 255.0).astype(np.int32)
        else:
            rgb_vals = np.clip(rgb_vals, 0, 255).astype(np.int32)
        colors = ['rgb({},{},{})'.format(int(r), int(g), int(b)) for r, g, b in rgb_vals]
        marker_dict = dict(size=3, color=colors, opacity=0.8)
    else:
        # Color by Z height (numeric -> use colorscale)
        colors = z
        marker_dict = dict(size=3, color=colors, colorscale='Viridis', opacity=0.8)

    trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=marker_dict
    )
    return trace

@app.route('/')
def index():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>Point Cloud Viewer</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body { font-family: sans-serif; margin: 0; padding: 20px; display: flex; flex-direction: column; height: 100vh; box-sizing: border-box; }
        #controls { margin-bottom: 20px; flex-shrink: 0; }
        #plot { width: 100%; flex-grow: 1; }
        button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
        input { padding: 10px; font-size: 16px; width: 80px; }
        .info { margin-left: 10px; font-weight: bold; }
    </style>
</head>
<body>
    <div id="controls">
        <button id="prevBtn">Previous (Left)</button>
        <span class="info">Frame: <input type="number" id="frameInput" value="0"> / <span id="totalFrames">0</span></span>
        <span class="info">Points: <span id="pointCount">0</span></span>
        <button id="nextBtn">Next (Right)</button>
    </div>
    <div id="plot"></div>

    <script>
        var currentFrame = 0;
        var totalFrames = {{ total_frames }};
        
        $('#totalFrames').text(totalFrames - 1);

        function loadFrame(index) {
            if (index < 0) index = 0;
            if (index >= totalFrames) index = totalFrames - 1;
            
            $.getJSON('/data/' + index, function(data) {
                var trace = data.data[0];
                var layout = data.layout;
                
                // Update point count
                var numPoints = 0;
                if (trace && trace.x) {
                    numPoints = trace.x.length;
                }
                $('#pointCount').text(numPoints);
                
                // Keep camera position if plot exists
                var gd = document.getElementById('plot');
                if (gd && gd.layout && gd.layout.scene && gd.layout.scene.camera) {
                    layout.scene.camera = gd.layout.scene.camera;
                }

                Plotly.react('plot', [trace], layout);
                
                currentFrame = index;
                $('#frameInput').val(currentFrame);
            });
        }

        $(document).ready(function() {
            loadFrame(0);

            $('#prevBtn').click(function() {
                loadFrame(currentFrame - 1);
            });

            $('#nextBtn').click(function() {
                loadFrame(currentFrame + 1);
            });

            $('#frameInput').change(function() {
                var val = parseInt($(this).val());
                loadFrame(val);
            });
            
            // Keyboard navigation
            $(document).keydown(function(e) {
                if (e.which == 37) { // left arrow
                   loadFrame(currentFrame - 1);
                } else if (e.which == 39) { // right arrow
                   loadFrame(currentFrame + 1);
                }
            });
        });
    </script>
</body>
</html>
    """, total_frames=len(point_clouds))

@app.route('/data/<int:index>')
def data(index):
    trace = get_trace(index)
    if trace is None:
        return jsonify({'error': 'Invalid index'}), 400
    
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        )
    )
    
    # Convert to JSON compatible format
    graph_json = json.dumps({
        'data': [trace], 
        'layout': layout
    }, cls=plotly.utils.PlotlyJSONEncoder)
    
    return graph_json

def main():
    global point_clouds
    parser = argparse.ArgumentParser(description="Visualize point clouds from a Zarr file.")
    parser.add_argument('--zarr_path', type=str, required=True, help='Path to the Zarr file (directory).')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on.')
    args = parser.parse_args()

    zarr_path = args.zarr_path
    if not os.path.exists(zarr_path):
        print(f"Error: Zarr path {zarr_path} does not exist.")
        return

    try:
        root = zarr.open(zarr_path, mode='r')
        if 'data' not in root or 'point_cloud' not in root['data']:
             print(f"Error: 'data/point_cloud' not found in {zarr_path}")
             return

        point_clouds = root['data']['point_cloud']
        print(f"Loaded point cloud dataset. Shape: {point_clouds.shape}")
        
        print(f"Starting server at http://127.0.0.1:{args.port}")
        app.run(host='0.0.0.0', port=args.port, debug=True, use_reloader=False)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

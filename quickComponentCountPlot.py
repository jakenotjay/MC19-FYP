import plotly.graph_objects as go

data = [98, 82, 60, 72, 58, 74, 121]
thresh = [1, 2, 5, 10, 15, 25, 50]

fig = go.Figure()
fig.add_trace(go.Scatter(x=thresh, y=data, mode='lines', line_shape='spline'))

# Update xaxis properties
fig.update_xaxes(title_text="Threshold Value")

# Update yaxis properties
fig.update_yaxes(title_text="Number of components detected")

# Update title and height
fig.update_layout(
    showlegend=False,
    # width=500,
    font=dict(
        size=16
    )
)

fig.show()
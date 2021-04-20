import pandas as pd
import plotly.graph_objects as go

filename = 'FinalFusedFull.pkl'
df = pd.read_pickle('./outputs/pkl/' + filename)

print(df)
print(df.dtypes)

noiseFig = go.Figure(data=go.Scatter(
    x=df['sliceN'], y=df['nNoisePixels'], 
    mode='lines',
    name='N of Noise Pixels (Isolated Pixels)'
    ))
noiseFig.update_layout(
    title='The number of noise pixels (isolated pixels) as a function of slice',
    xaxis_title='Slice Number',
    yaxis_title='Number of Noise pixels',
    legend_title='',
    font=dict(
        # family="Courier New, monospace",
        size=18,
        # color="RebeccaPurple"
    )
)
noiseFig.show()

noiseFracFig = go.Figure(data=go.Scatter(
    x=df['sliceN'], y=df['noiseFracSlice'], 
    mode='lines',
    name='Fraction of noise pixels/total non zero pixels'
    ))
noiseFracFig.update_layout(
    title='The fraction of noise pixels/total non zero pixels as a function of slice',
    xaxis_title='Slice Number',
    yaxis_title='Fraction of noise pixels/total non zero pixels',
    legend_title='',
    font=dict(
        # family="Courier New, monospace",
        size=18,
        # color="RebeccaPurple"
    )
)
noiseFracFig.show()

meanNoiseIntensityFig = go.Figure(data=go.Scatter(
    x=df['sliceN'], y=df['meanNoiseIntensity'], 
    mode='lines',
    name='Mean intensity of noise pixels'
    ))
meanNoiseIntensityFig.update_layout(
    title='Mean intensity of noise pixels (isolated pixels)',
    xaxis_title='Slice Number',
    yaxis_title='Mean Intensity (0-255)',
    legend_title='',
    font=dict(
        # family="Courier New, monospace",
        size=18,
        # color="RebeccaPurple"
    )
)
meanNoiseIntensityFig.show()

nonZeroPixelFracSliceFig = go.Figure(data=go.Scatter(
    x=df['sliceN'], y=df['nonZeroPixelFracSlice'], 
    mode='lines',
    name='Non zero pixel fraction'
    ))
nonZeroPixelFracSliceFig.update_layout(
    title='The fraction of non-zero pixels to image resolution',
    xaxis_title='Slice Number',
    yaxis_title='Non zero pixel fraction',
    legend_title='',
    font=dict(
        # family="Courier New, monospace",
        size=18,
        # color="RebeccaPurple"
    )
)
nonZeroPixelFracSliceFig.show()

nonZeroPixelsSliceFig = go.Figure(data=go.Scatter(
    x=df['sliceN'], y=df['nonZeroPixelsSlice'], 
    mode='lines',
    name='Number of non zero pixels'
    ))
nonZeroPixelsSliceFig.update_layout(
    title='Number of non zero pixels as a function fo slice number',
    xaxis_title='Slice Number',
    yaxis_title='Number of non zero pixels',
    legend_title='',
    font=dict(
        # family="Courier New, monospace",
        size=18,
        # color="RebeccaPurple"
    )
)
nonZeroPixelsSliceFig.show()
import plotly.express as px

params= [11173962,308826,53082]
accuracy =[95.72,88.05,91.01]
names= ['Big ResNet18','Distillation','Pruning & Fine tuning']

fig = px.scatter(x=params,y=accuracy, text=names, log_x=True, size_max=60)

fig.update_traces(textposition='top center')

fig.update_layout(height=400, width=1400,title_text='Accuracy vs number of parameters ',xaxis_title="Number of parameters", yaxis_title="Accuracy")

fig.show()
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from keras.utils import pad_sequences
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from keras import losses, optimizers, metrics
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import keras
from torch_geometric_temporal.nn.recurrent import A3TGCN
from torch_geometric_temporal.signal import temporal_signal_split
from model import * 

app = Flask(__name__)

#Timeseries model (Model1)
model1 = keras.models.load_model("models\Timeseries_model.h5", compile=False)
dataset1 = pd.read_csv(r"models\unit_tabular.csv", parse_dates=True)


#GNN model (Model2)
class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features,
                           out_channels=32,
                           periods=periods)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h

model2 = TemporalGNN(node_features=2, periods=12)
model2.load_state_dict(torch.load("models\A3TGCN_model.pt"))
model2.eval()
device = torch.device('cpu')

dataset2 = torch.load("models\dataset.pt")

#Combined model
combinedmodel = CombinedModel(model1, model2)
test_generator, test_dataset, truths = combinedmodel.get_test(dataset1, dataset2)
predictions, labels = combinedmodel.predict(test_generator, test_dataset)

#Dictionary for other purpose
suburb_stat = np.load("models\suburb_stat.npy",allow_pickle='TRUE').item()
suburb_hm = {0: "Brisbane City", 1: "New Farm", 2: "Teneriffe", 3: "Newstead", 4: "Fortitude Valley", 5: "Bowen Hills", 6: "Herston", 7: "Kelvin Groove", 8: "Red Hill", 9: "Paddington", 10: "Milton", 11: "South brisbane"}

sensor = -1

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def statistic():
    global sensor
    form = request.form.to_dict(flat=False)
    sensor = int(form['node'][0])
    
    truth_snapshot = truths[sensor]
    true = []
    for num in truth_snapshot:
        true.append(num)
  
    true = np.array(true)
    values = (true * suburb_stat[sensor][1] + suburb_stat[sensor][0])/1000
    values = values.tolist()
    timeline = [n for n in range(len(true))]
    suburb = suburb_hm[sensor]
    
    return render_template("index.html", suburb=suburb, timeline=timeline, values=values)

def get_statistic():
    global sensor
    
    truth_snapshot = truths[sensor]
    true = []
    for num in truth_snapshot:
        true.append(num)
  
    true = np.array(true)
    values = (true * suburb_stat[sensor][1] + suburb_stat[sensor][0])/1000
    values = values.tolist()
    timeline = [n for n in range(len(true))]
    suburb = suburb_hm[sensor]
    
    return suburb, timeline, values

@app.route('/predict', methods=['POST'])
def predict():
    #form = request.form.to_dict(flat=False)
    #sensor = int(form['node'][0])

    pred_snapshot = [pred[sensor].detach().cpu().numpy() for pred in predictions]
    preds = []
    for n in pred_snapshot:
      for num in n:
        preds.append(num.item())
    
    preds = np.array(preds)
    
    #Scale is 1000
    prediction = ((preds) * suburb_stat[sensor][1] + suburb_stat[sensor][0])/1000
    prediction = prediction.tolist()
    
    
    label_snapshot = [label[sensor].cpu().numpy() for label in labels]
    labs = []
    for n in label_snapshot:
      for num in n:
        labs.append(num.item())
    
    labs = np.array(labs)
    labs = (labs * suburb_stat[sensor][1] + suburb_stat[sensor][0])/1000
    labs = labs.tolist()
    predict_timeline = [n for n in range(len(preds))]
    
    suburb, timeline, values = get_statistic()
    
    return render_template("index.html", predict_timeline=predict_timeline, prediction=prediction, labs=labs,
                           suburb=suburb, timeline=timeline, values=values)

if __name__ == "__main__":
    app.run()
    
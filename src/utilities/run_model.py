
from utilities.translation_pipeline import Pipe
from utilities.loaders import load_reviews, load_w2vec_model
from utilities.lstm import LSTM_test, loss_calc


def run(data, language, language_short, lang_model, num_reviews):

    shuf_ge_df = ge_df.sample(frac=1)
    target = shuf_ge_df['sentiment'].to_numpy()[:100]
    target = torch.from_numpy(target).float().reshape(-1,1)

    td = TensorDataset(torch_data, target)
    dl = DataLoader(td, batch_size=20, shuffle=True)

    model = LSTM_test()

    loss = loss_calc(100, td, dl)
    print(loss)

    pred = torch.round(model(torch_data[:100])[0])
    acc = sum(pred == target[:100]) / 100
    print(acc)
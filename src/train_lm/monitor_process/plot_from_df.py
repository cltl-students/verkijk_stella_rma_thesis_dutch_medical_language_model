###
@author StellaVerkijk
This script orders the dataframes created by log_to_df.py and plots the learning rate and loss over pre-training time steps
###

import pandas as pd
import matplotlib.pyplot as plt
import click
       
@click.command()
@click.argument("file_to_plot")
@click.argument("outdir")
@click.option(
    "--plot-loss/--no-plot-loss", default=True
)
@click.option(
    "--plot-lr/--no-plot-lr", default=False
)

def main(file_to_plot: str, outdir: str, plot_loss: bool, plot_lr: bool):
    
    
    df = pd.read_csv(file_to_plot, header = 0, index_col = None, sep = ',', encoding = 'utf-8')

    loss = []
    steps = []
    data = []


    for index, row in df.iterrows():
        if df.iloc[index]['metric'] == 'train/loss':
            #if df.iloc[index]['step'] <= 10000:
            steps.append(df.iloc[index]['step'])
            loss.append(df.iloc[index]['value'])
            #else:
            #    break
            data.append(['loss', df.iloc[index]['step'], df.iloc[index]['value']])
                
    df_loss = pd.DataFrame()
    df_loss['step'] = steps
    df_loss['loss'] = loss


    lr = []
    steps_lr = []

    for index, row in df.iterrows():
        if df.iloc[index]['metric'] == 'train/learning_rate':
            #if df.iloc[index]['step'] <= 10000:
            steps_lr.append(df.iloc[index]['step'])
            lr.append(df.iloc[index]['value'])
            #else:
            #    break
            
    #print(i)       
    df_lr = pd.DataFrame()
    df_lr['step'] = steps
    df_lr['lr'] = lr

    eval_loss = []
    steps_eval = []

    for index, row in df.iterrows():
        if df.iloc[index]['metric'] == 'eval/loss':
            #if df.iloc[index]['step'] <= 10000:
            steps_eval.append(df.iloc[index]['step'])
            eval_loss.append(df.iloc[index]['value'])
            #else:
            #    break
            data.append(['eval_loss', df.iloc[index]['step'], df.iloc[index]['value']])


    df_complete = pd.DataFrame(data, columns = ['metric', 'step', 'value'])

    df_final = df_complete.pivot(index = 'step', columns = 'metric', values = 'value')

    if plot_loss:
        plt.figure()
        plt.plot(steps_eval, eval_loss, label = "eval")
        plt.plot(steps, loss, label = "train")
        plt.legend()
        plt.grid()
        plt.xlabel("steps")
        plt.ylabel("loss")
        plt.savefig(outdir)
        plt.show()
    
    if plot_lr:
        df_lr.plot(x = 'step', y='lr', kind='line')
        plt.legend()
        plt.grid()
        plt.savefig(outdir)
        plt.show()


if __name__ == "__main__":
    main()


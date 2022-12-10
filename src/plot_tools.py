# plot_tools.py
import os
import matplotlib.pyplot as plt
import seaborn as sb

def set_paras(x_title,y_title,title=None,filename=None,file_dir='plots',has_label=False):

    '''set all the parameters in the figure and save files'''
    if has_label:
        plt.legend()
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)

    if filename:
        full_path = os.path.join(file_dir, filename)
        plt.savefig(full_path)
        plt.close()
        # plt.show() #for testing
    else:
        plt.show()

def make_dir(file_dir):
    '''checks if the directory exists if not make one'''
    if file_dir:
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)

def plot_2D(x, y, plot_count=1,title=None,x_title=None,y_title=None,label=False,filename=None,
        file_dir='plots', multi_x=True):

    '''plots inputs: x:array like of array like, y:array like of array likes,
    plot_count:int(number of plots),title:string, file_dir:string,colour:string'''

    make_dir(file_dir)

    if plot_count == 1:
        x = [x]
        y = [y]

    for i in range(plot_count):
        if multi_x:
            if label:
                plt.plot(x[i],y[i],label=label[i])
            else:
                plt.plot(x[i],y[i])
        else:
            if label:
                plt.plot(x,y[i],label=label[i])
            else:
                plt.plot(x,y[i])


    set_paras(x_title, y_title, title, filename, file_dir, label)


def correlation(x):
    return

def confusion(y_test, y_pred):
    return

def show_data():
    return
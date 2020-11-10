from matplotlib import pyplot
from matplotlib.colors import ListedColormap
import numpy

def drawClasses(X_set, y_set ,c1, c2, title_name, x_label, y_label, classes, classifier=None):
    pyplot.figure()
#    X_set, y_set = X_train, y_train
    X1, X2 = numpy.meshgrid(numpy.arange(start = X_set[:, c1].min() - 1, stop = X_set[:, c1].max() + 1, step = 0.01),
                         numpy.arange(start = X_set[:, c2].min() - 1, stop = X_set[:, c2].max() + 1, step = 0.01))
    if classifier is not None:
        pyplot.contourf(X1, X2, classifier.predict(numpy.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.25, cmap = ListedColormap(classes))
    pyplot.xlim(X1.min(), X1.max())
    pyplot.ylim(X2.min(), X2.max())
    for i, j in enumerate(numpy.unique(y_set)):
        pyplot.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(classes)(i), label = j)
    pyplot.title('Classifier ({})'.format(title_name))
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
    pyplot.legend()
    pyplot.show()

def draw_classes(X, y, columns, graph_title, classes_color, model=None):
    pyplot.figure()

    min_column1 = X[columns[0]].min() if X[columns[0]].min() != 0 else X[columns[0]].min() - 1
    max_column1 = X[columns[0]].max() + 1
    min_column2 = X[columns[1]].min() if X[columns[1]].min() != 0 else X[columns[1]].min() - 1
    max_column2 = X[columns[1]].max() + 1

    X1, X2 = numpy.meshgrid(
        numpy.arange(start=min_column1, stop=max_column1, step=0.01),
        numpy.arange(start=min_column2, stop=max_column2, step=0.01)
    )
    if model is not None:
        pyplot.contourf(X1, X2,
                        model.predict(numpy.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                        alpha=0.25, cmap=ListedColormap(classes_color))

    # Draw the graph
    pyplot.xlim(min_column1, max_column1)
    pyplot.ylim(min_column2, max_column2)

    for index, value in enumerate(numpy.unique(y)):
        pyplot.scatter(X[y == value][columns[0]], X[y == value][columns[1]],
                    c=ListedColormap(classes_color)(index), label = value)

    pyplot.title(graph_title)
    pyplot.xlabel(columns[0])
    pyplot.ylabel(columns[1])
    pyplot.legend()
    pyplot.show()

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

def draw_classes(X, y, columns, graph_title, classes_color, model=None, resolution=0.5):
    pyplot.figure()

    min_column1 = X[columns[0]].min() if X[columns[0]].min() != 0 else X[columns[0]].min() - 1
    max_column1 = X[columns[0]].max() + 1
    min_column2 = X[columns[1]].min() if X[columns[1]].min() != 0 else X[columns[1]].min() - 1
    max_column2 = X[columns[1]].max() + 1

    X1, X2 = numpy.meshgrid(
        numpy.arange(start=min_column1, stop=max_column1, step=resolution),
        numpy.arange(start=min_column2, stop=max_column2, step=resolution)
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

def draw_classes_regions(X, y, model, features, test_index=None, resolution=0.5):
    # Setup markers and colors
    markers = ('s', 'x', 'o', '^', 'v', 'i')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    color_map = ListedColormap(colors[:len(numpy.unique(y))])

    # Prepare the coordinates array
    X1_min, X1_max = X[features[0]].min() - 1, X[features[0]].max() + 1
    X2_min, X2_max = X[features[1]].min() - 1, X[features[1]].max() + 1
    meshgrid1, meshgrid2 = numpy.meshgrid(numpy.arange(X1_min, X1_max, resolution),
                                          numpy.arange(X2_min, X2_max, resolution))

    # Prediction
    prediction = model.predict(numpy.array([meshgrid1.ravel(), meshgrid2.ravel()]).T)
    prediction = prediction.reshape(meshgrid1.shape)

    # Graph preparation
    pyplot.contourf(meshgrid1, meshgrid2, prediction, alpha=0.3, cmap=color_map)
    pyplot.xlim(meshgrid1.min(), meshgrid1.max())
    pyplot.ylim(meshgrid2.min(), meshgrid2.max())

    for index, value in enumerate(numpy.unique(y)):
        pyplot.scatter(x=X[y == value][features[0]],
                       y=X[y == value][features[1]],
                       alpha=0.8,
                       c=colors[index],
                       marker=markers[index],
                       label=value,
                       edgecolors='black')

    # Highlight test sample
    if test_index:
        X_test, y_test = X[test_index][:], y[test_index]
        
        pyplot.scatter(X_test[:][0],
                       X_test[:][1],
                       c='',
                       edgecolors='black',
                       alpha=1.0,
                       linewidths=1,
                       marker='o',
                       s=100,
                       label='test set')
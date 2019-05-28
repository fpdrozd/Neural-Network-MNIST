import numpy
import matplotlib.pyplot as plt

testing_data_file = open('mnist_dataset/mnist_test_10.csv', 'r')
testing_data_list = testing_data_file.readlines()
testing_data_file.close()

for record in testing_data_list:
    all_values = record.split(',')
    image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.show()

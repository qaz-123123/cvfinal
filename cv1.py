import numpy as np
import copy
import matplotlib.pyplot as plt


# turn the images in train dataset and test dataset to a two-dimensional matrix from a one-dimensional array
def turn_to_matrix(X, row_num, col_num):
    x_matrix = np.zeros((row_num, col_num), dtype=np.uint16)
    for row in range(row_num):
        for col in range(col_num):
            x_matrix[row, col] = X[row*col_num+col]

    return x_matrix


def activation_function(func, f_x):
    """
    :param func: this parameter can be a str or a function, str represents the kind of activation function
    :param f_x: the independent variable of the activation function
    :return: the output of the activation function
    """

    if isinstance(func, str):
        if func == 'sigmoid':
            return 1 / (1+np.exp(-f_x))
        if func == 'relu':
            return np.maximum(0, f_x)
        if func == 'tanh':
            return np.tanh(f_x)
        if func == 'softmax':
            return np.exp(f_x-np.max(f_x)) / np.sum(np.exp(f_x-np.max(f_x)))
    else:
        return func(f_x)


# this function achieve one time convolution operate between two matrices with the same size
def cal_conv(f_x, f_y, is_rotation, *bias):
    """
    :param f_x: a matrix
    :param f_y: another matrix with same shape as x
    :param is_rotation: boolean, if True the matrix y is rotated 180 degrees, else it is not rotated
    :param bias: maybe exists
    :return: a float, the result of convolution if is_rotation=True or matrix inner product if is_rotation=False
    """
    result = 0
    for matrix_row in range(f_x.shape[0]):
        for matrix_col in range(f_x.shape[1]):
            if is_rotation:
                result += f_x[matrix_row, matrix_col] * f_y[f_x.shape[0]-matrix_row-1, f_x.shape[1]-matrix_col-1]
            else:
                result += f_x[matrix_row, matrix_col] * f_y[matrix_row, matrix_col]

    result = result + bias[0] if len(bias) else result
    return result


class ConvolutionLayer:

    def __init__(self, input_im, f_input_channel_num, f_padding_size, f_conv_kernel_size, f_conv_kernel, f_conv_bias, f_conv_stride_size, f_activation_func, f_pooling_type, f_pooling_kernel_size, f_pooling_stride_size):
        """
        :param input_im: a list storing matrices with the number equal to input channel number
        :param f_input_channel_num: an int, input channel number
        :param f_padding_size: a list, storing two integers indicating row and column number for extra elements
        :param f_conv_kernel_size: a list, storing two integers indicating kernel size for convolution
        :param f_activation_func: a str or a function, activation function
        :param f_pooling_type: a str, pooling type for pooling layer
        :param f_conv_kernel: a nested list storing lists for kernels. Every element of outer list is a kernel tensor. If input channel
                            number is 1 then kernel location of inner list will be one matrix and if input channel number is larger than 1
                            then this location will be a list whose length is equal to input channel number.
        :param f_conv_bias: a list storing bias matrices used in convolution, the number of them is equal to output_channel_num
        :param f_conv_stride_size: a list storing two integers indicating the stride size for row and col when doing convolution
        :param f_pooling_kernel_size: a list storing two integers indicating the row and column size of the kernel used in pooling
        :param f_pooling_stride_size: a list storing two integers indicating the row and column size of the stride used in pooling
        """
        self.input_im = input_im
        self.input_channel_num = f_input_channel_num

        if self.input_channel_num == 1:
            self.input_size = self.input_im.shape
        else:
            self.input_size = self.input_im[0].shape

        self.activation_func = f_activation_func
        self.padding_size = f_padding_size
        self.conv_kernel_size = f_conv_kernel_size
        self.conv_stride_size = f_conv_stride_size
        self.pooling_type = f_pooling_type
        self.pooling_kernel_size = f_pooling_kernel_size
        self.pooling_stride_size = f_pooling_stride_size
        self.conv_kernel = f_conv_kernel
        self.conv_bias = f_conv_bias
        self.output_channel_num = len(self.conv_kernel)

    def padding(self):
        """
        # :param padding_size: a list storing two integers indicating row and column number for extra elements
        :return: padding result, a list storing padding result for each channel of the input image
        """

        # get the height and width of the input image
        h, w = self.input_size[0], self.input_size[1]

        # padding, using zeros for extra elements
        padding_list = []
        for f_i in range(self.input_channel_num):
            padding_x = np.zeros((h+2*self.padding_size[0], w+2*self.padding_size[1]), dtype=np.uint16)
            padding_x[self.padding_size[0]:(h+self.padding_size[0]), self.padding_size[1]:(w+self.padding_size[1])] = self.input_im[f_i]
            padding_list.append(padding_x)

        return padding_list

    def convolution(self):
        """
        # :param stride_size: a list storing two integers indicating the stride size for row and col when doing convolution
        # :param padding_size: a list storing two integers indicating row and column number for extra elements
        # :param bias_matrix: a list storing bias matrices used in convolution, the number of them is equal to output_channel_num
        # :param kernel_info: a nested list storing lists for kernels. Every element of outer list is a kernel tensor. If input channel
        #                     number is 1 then kernel location of inner list will be one matrix and if input channel number is larger than 1
        #                     then this location will be a list whose length is equal to input channel number.
        :return: convolution result, a list storing the result of convolution operation between each kernel and input image after activating
        """

        # get padding result, used to do convolution
        padding_list = self.padding()

        # used to store convolution result
        conv_result = []

        # each 'f_i' represents a kernel used in convolution and each cycle gets a convolution result between this kernel and padding image
        for f_i in range(self.output_channel_num):
            # use input image size, padding size, kernel size and stride size to get convolution result matrix size
            conv_result_row_num = (self.input_size[0]+2*self.padding_size[0]-self.conv_kernel_size[0]) // self.conv_stride_size[0] + 1
            conv_result_col_num = (self.input_size[1]+2*self.padding_size[1]-self.conv_kernel_size[1]) // self.conv_stride_size[1] + 1
            conv_result_matrix = np.zeros((conv_result_row_num, conv_result_col_num), dtype=np.float64)

            # calculate convolution result by element
            conv_result_row_index = 0
            for row in range(0, padding_list[0].shape[0], self.conv_stride_size[0]):
                conv_result_col_index = 0
                for col in range(0, padding_list[0].shape[1], self.conv_stride_size[1]):
                    # to get an element of result, calculate the range of padding image used to do convolution
                    conv_row_right = row + self.conv_kernel_size[0]
                    conv_col_bottom = col + self.conv_kernel_size[1]

                    # what kernel covers in padding image may consist of the part inside the range of it and part out of it
                    # if this condition occurs in column perspective, then give up the rest part of padding image and change to next row
                    if conv_row_right > self.input_size[0] + 2*self.padding_size[0]:
                        continue
                    # if this condition occurs in row perspective, then give up the rest part of padding image and stop cycle
                    if conv_col_bottom > self.input_size[1] + 2*self.padding_size[1]:
                        break

                    # calculate convolution by channel, if there are more than one channel, add the result of each channel
                    if self.input_channel_num == 1:
                        conv_result_matrix[conv_result_row_index, conv_result_col_index] += cal_conv(padding_list[0][row:conv_row_right, col:conv_col_bottom], self.conv_kernel[f_i][0], False)
                    else:
                        for channel in range(self.input_channel_num):
                            conv_result_matrix[conv_result_row_index, conv_result_col_index] += cal_conv(padding_list[channel][row:conv_row_right, col:conv_col_bottom], self.conv_kernel[f_i][channel], False)
                    # add bias
                    conv_result_matrix[conv_result_row_index, conv_result_col_index] += self.conv_bias[f_i][conv_result_row_index, conv_result_col_index]
                    conv_result_col_index += 1
                conv_result_row_index += 1

            conv_result.append(conv_result_matrix)

        return conv_result

    def conv_activate(self, conv_result):
        """
        :param conv_result: a list storing the result of convolution operation
        :return: the result of activation
        """

        # calculate the result of activation by element by channel
        for f_i in range(self.output_channel_num):
            for row in range(conv_result[0].shape[0]):
                for col in range(conv_result[0].shape[1]):
                    conv_result[f_i][row][col] = activation_function(self.activation_func, conv_result[f_i][row, col])

        return conv_result

    def pooling(self, input_matrix):
        """
        # :param kernel_size: a list storing two integers indicating the row and column size of the kernel used in pooling
        # :param stride_size: a list storing two integers indicating the row and column size of the stride used in pooling
        :param input_matrix: a matrix if channel_num is 1 or a list storing matrices if channel_num is larger than 1
        :return: a list storing result of pooling, the number of it may be one or more, which depends on channel number, and the gradient of pooling
        """

        # get the height and width of input matrix
        h, w = input_matrix[0].shape[0], input_matrix[0].shape[1]

        # calculate the size of pooling result
        pooling_size = (int(np.ceil((h-self.pooling_kernel_size[0])/self.pooling_stride_size[0])+1), int(np.ceil((w-self.pooling_kernel_size[1])/self.pooling_stride_size[1])+1))
        # used to store the pooling result of each channel
        pooling_result = []
        # used to store the pooling gradient of each channel
        pooling_gradient = []

        # calculate pooling result for each channel
        for channel in range(self.output_channel_num):
            pooling_matrix = np.zeros(pooling_size, dtype=np.float64)
            # stores the gradient value of each element
            pooling_gradient_matrix = np.zeros((h, w), dtype=np.float64)

            # calculate result by element
            pooling_matrix_row_index = 0
            for row in range(0, h, self.pooling_stride_size[0]):
                pooling_matrix_col_index = 0
                for col in range(0, w, self.pooling_stride_size[1]):
                    # what kernel covers in input matrix may consist of both inside and outside the range of matrix
                    # regardless of row or column respective, still do pooling for the rest part in the matrix
                    if row + self.pooling_kernel_size[0] > h:
                        row_right = h
                    else:
                        row_right = row + self.pooling_kernel_size[0]
                    if col + self.pooling_kernel_size[1] > w:
                        col_bottom = w
                    else:
                        col_bottom = col + self.pooling_kernel_size[1]

                    # calculate the result by either max-pooling or mean-pooling
                    used_to_pooling = input_matrix[channel][row:row_right, col:col_bottom]
                    if self.pooling_type == 'max':
                        pooling_matrix[pooling_matrix_row_index, pooling_matrix_col_index] = np.max(used_to_pooling)
                        # record the location of max value
                        max_index = np.where(used_to_pooling == pooling_matrix[pooling_matrix_row_index, pooling_matrix_col_index])
                        pooling_gradient_matrix[max_index[0][0]+row, max_index[1][0]+col] = 1
                    if self.pooling_type == 'mean':
                        pooling_matrix[pooling_matrix_row_index, pooling_matrix_col_index] = np.mean(used_to_pooling)
                        # calculate the weight of mean operation
                        mean_weight = 1 / ((row_right - row)*(col_bottom - col))
                        pooling_gradient_matrix[row:row_right, col:col_bottom] = mean_weight * np.ones((row_right-row, col_bottom-col), dtype=np.float64)

                    pooling_matrix_col_index += 1

                pooling_matrix_row_index += 1

            pooling_result.append(pooling_matrix)
            pooling_gradient.append(pooling_gradient_matrix)

        return pooling_result, pooling_gradient

    def conv_gradient_backpropagation(self, d_loss_over_output, pooling_gradient, activation_result, padding_list, *grad_af_over_conv):
        """
        :param d_loss_over_output: \frac {\partial loss}{\partial pooling_output}
        :param pooling_gradient: a list storing the gradient matrix for each channel of pooling
        # :param pooling_kernel_size: the size of kernel used in pooling
        # :param pooling_stride_size: the stride size used in pooling
        :param activation_result: the result of activation function
        # :param conv_stride_size: the stride size used in convolution operation
        :param padding_list: the result of padding
        # :param conv_kernel: the kernel matrices used in convolution operation
        :param grad_af_over_conv: the gradient of activation function which is calculated in advance
        :return: \frac {\partial loss}{\partial pooling_input}(that is \frac {\partial loss}{\partial af_result}), \frac {\partial loss}{\partial convolution_result},
                 \frac {\partial loss}{\partial kernel_weight}, \frac {\partial loss}{\partial bias_matrix}, \frac {\partial loss}{\partial padding_result},
                 \frac {\partial loss}{\partial padding_input}(that is \frac {\partial loss}{\partial input_of_conv_layer})
        """
        # get the size of matrix before pooling, which is equal to the size of pooling gradient matrix
        pooling_gradient_matrix_h, pooling_gradient_matrix_w = pooling_gradient[0].shape[0], pooling_gradient[0].shape[1]
        # stores the gradient matrices of loss with respect to input of pooling, that is result of activation function, for each channel
        d_loss_over_pooling_input = []
        # calculate gradient matrix by channel
        for channel in range(self.output_channel_num):
            # stores the gradient matrix for this channel
            d_loss_over_pooling_input_matrix = np.zeros((pooling_gradient_matrix_h, pooling_gradient_matrix_w), dtype=np.float64)
            # get the pooling gradient matrix for this channel
            pooling_gradient_matrix = pooling_gradient[channel]
            # simulate the double-layer cycle as pooling operation to calculate gradient of each element
            d_loss_over_pooling_output_matrix_row_index = 0
            for row in range(0, pooling_gradient_matrix_h, self.pooling_stride_size[0]):
                d_loss_over_pooling_output_matrix_col_index = 0
                for col in range(0, pooling_gradient_matrix_w, self.pooling_stride_size[1]):
                    if row + self.pooling_kernel_size[0] > pooling_gradient_matrix_h:
                        row_right = pooling_gradient_matrix_h
                    else:
                        row_right = row + self.pooling_kernel_size[0]
                    if col + self.pooling_kernel_size[1] > pooling_gradient_matrix_w:
                        col_bottom = pooling_gradient_matrix_w
                    else:
                        col_bottom = col + self.pooling_kernel_size[1]

                    if self.pooling_type == 'max':
                        # find the location of indicator 1 to the max value
                        max_index = np.where(pooling_gradient_matrix[row:row_right, col:col_bottom] == 1)
                        # calculate the gradient value of this element, that is multiplying 1 with the gradient for next step
                        d_loss_over_pooling_input_matrix[max_index[0][0]+row, max_index[1][0]+col] = d_loss_over_output[channel][d_loss_over_pooling_output_matrix_row_index, d_loss_over_pooling_output_matrix_col_index]
                    if self.pooling_type == 'mean':
                        # calculate the gradient value of this element, that is multiplying mean weight with the gradient for next step
                        pooling_gradient_matrix[row:row_right, col:col_bottom] *= d_loss_over_output[channel][d_loss_over_pooling_output_matrix_row_index, d_loss_over_pooling_output_matrix_col_index]

                    d_loss_over_pooling_output_matrix_col_index += 1

                d_loss_over_pooling_output_matrix_row_index += 1

            d_loss_over_pooling_input.append(d_loss_over_pooling_input_matrix)

        # stores the gradient matrices of the loss with respect to convolution result for each channel
        d_loss_over_conv = []
        # calculate the gradient matrix by channel
        for channel in range(self.output_channel_num):
            # stores the gradient matrix for this channel, the size of which is equal to the size of activation result with the same size as pooling gradient matrix of this channel
            d_loss_over_conv_matrix = np.zeros((pooling_gradient_matrix_h, pooling_gradient_matrix_w), dtype=np.float64)
            # an all-ones matrix, used to calculate gradient matrix
            all_ones_matrix = np.ones((pooling_gradient_matrix_h, pooling_gradient_matrix_w), dtype=np.float64)
            # if the type of self.activation_func is str, then calculate the gradient of it, otherwise calculate it in advance
            # here we calculate the gradient by element first, then construct the results to matrix form
            if isinstance(self.activation_func, str):
                if self.activation_func == 'relu':
                    d_af_over_conv = np.where(activation_result[channel] < 0, 0, 1)
                    d_loss_over_conv_matrix = np.multiply(d_loss_over_pooling_input[channel], d_af_over_conv)
                if self.activation_func == 'sigmoid':
                    d_loss_over_conv_matrix = np.multiply(d_loss_over_pooling_input[channel], np.multiply(activation_result[channel], all_ones_matrix - activation_result[channel]))
                if self.activation_func == 'tanh':
                    d_loss_over_conv_matrix = np.multiply(d_loss_over_pooling_input[channel], all_ones_matrix-np.multiply(activation_result[channel], activation_result[channel]))
                if self.activation_func == 'softmax':
                    for row_of_linear in range(pooling_gradient_matrix_h):
                        for col_of_linear in range(pooling_gradient_matrix_w):
                            s = 0
                            for row_of_af in range(pooling_gradient_matrix_h):
                                for col_of_af in range(pooling_gradient_matrix_w):
                                    if row_of_linear == row_of_af and col_of_linear == col_of_af:
                                        s += d_loss_over_pooling_input[channel][row_of_linear, col_of_linear] * activation_result[channel][row_of_linear, col_of_linear] * (1 - activation_result[channel][row_of_linear, col_of_linear])
                                    else:
                                        s -= d_loss_over_pooling_input[channel][row_of_linear, col_of_linear] * activation_result[channel][row_of_linear, col_of_linear] * activation_result[channel][row_of_af, col_of_af]

                            d_loss_over_conv_matrix[row_of_linear, col_of_linear] = s

                d_loss_over_conv.append(d_loss_over_conv_matrix)
            else:
                d_loss_over_conv.append(grad_af_over_conv[channel])

        # stores the gradient matrices of loss with respect to kernel matrix for each output channel, which is a list of matrices with number equal to input channel number
        d_loss_over_kernel = []
        # calculate gradient by output channel, that is gradient for kernel tensor
        for output_channel in range(self.output_channel_num):
            # stores kernel gradient for this channel
            kernel_gradient_for_each_channel = []
            # used to calculate gradient
            d_loss_over_conv_matrix = d_loss_over_conv[output_channel]
            # calculate gradient of each kernel matrix in kernel tensor by input channel, a list of kernel matrices
            for input_channel in range(self.input_channel_num):
                # used to calculate gradient
                padding_input_matrix = padding_list[input_channel]
                # calculate the size of gradient matrix
                gradient_size_h = (padding_input_matrix.shape[0]-d_loss_over_conv_matrix.shape[0]) // self.conv_stride_size[0] + 1
                gradient_size_w = (padding_input_matrix.shape[1]-d_loss_over_conv_matrix.shape[1]) // self.conv_stride_size[1] + 1
                # stores gradient
                gradient_matrix = np.zeros((gradient_size_h, gradient_size_w), dtype=np.float64)
                # calculate gradient matrix by element
                gradient_matrix_row_index = 0
                for row in range(0, padding_input_matrix.shape[0], self.conv_stride_size[0]):
                    gradient_matrix_col_index = 0
                    for col in range(0, padding_input_matrix.shape[1], self.conv_stride_size[1]):
                        # to get an element of result, calculate the range of padding input matrix used to do convolution
                        conv_row_right = row + d_loss_over_conv_matrix.shape[0]
                        conv_col_bottom = col + d_loss_over_conv_matrix.shape[1]

                        # what kernel covers in padding input matrix may consist of the part inside the range of it and part out of it
                        # if this condition occurs in column perspective, then give up the rest part of the padding input matrix and change to next row
                        if conv_row_right > padding_input_matrix.shape[0]:
                            continue
                        # if this condition occurs in row perspective, then give up the rest part of padding input matrix and stop cycle
                        if conv_col_bottom > padding_input_matrix.shape[1]:
                            break

                        # calculate convolution as a result of gradient
                        gradient_matrix[gradient_matrix_row_index, gradient_matrix_col_index] = cal_conv(padding_input_matrix[row:conv_row_right, col:conv_col_bottom], d_loss_over_conv_matrix, False)
                        gradient_matrix_col_index += 1

                    gradient_matrix_row_index += 1

                kernel_gradient_for_each_channel.append(gradient_matrix)

            d_loss_over_kernel.append(kernel_gradient_for_each_channel)

        # the gradient matrices of loss with respect to bias matrix for each output channel equal to the gradient of loss w.r.t convolution
        d_loss_over_bias = copy.deepcopy(d_loss_over_conv)

        # reorganize the nested list stores kernels used to calculate forward convolution, making it be able to calculate gradient
        reorganize_conv_kernel = [[self.conv_kernel[f_i][f_j] for f_i in range(self.output_channel_num)] for f_j in range(self.input_channel_num)]
        # stores the gradient matrices of loss with respect to padding output
        d_loss_over_padding_output = []
        # calculate gradient matrix by channel
        for input_channel in range(self.input_channel_num):
            # stores the gradient matrix for this channel
            d_loss_over_padding_output_matrix = np.zeros(padding_list[input_channel].shape, dtype=np.float64)
            # calculate gradient by element
            d_loss_over_padding_output_matrix_row_index = 0
            for row in range(0, pooling_gradient_matrix_h, self.conv_stride_size[0]):
                d_loss_over_padding_output_matrix_col_index = 0
                for col in range(0, pooling_gradient_matrix_w, self.conv_stride_size[1]):
                    # to get an element of result, calculate the range of matrix d_loss_over_conv used to do convolution
                    conv_row_right = row + self.conv_kernel_size[0]
                    conv_col_bottom = col + self.conv_kernel_size[1]

                    # what kernel covers in matrix d_loss_over_conv may consist of the part inside the range of it and part out of it
                    # if this condition occurs in column perspective, then give up the rest part of matrix d_loss_over_conv and change to next row
                    if conv_row_right > pooling_gradient_matrix_h:
                        continue
                    # if this condition occurs in row perspective, then give up the rest part of matrix d_loss_over_conv and stop cycle
                    if conv_col_bottom > pooling_gradient_matrix_w:
                        break

                    # calculate gradient value by convolution
                    for output_channel in range(self.output_channel_num):
                        d_loss_over_padding_output_matrix[d_loss_over_padding_output_matrix_row_index, d_loss_over_pooling_output_matrix_col_index] += cal_conv(d_loss_over_conv[output_channel][row:conv_row_right, col:conv_col_bottom], reorganize_conv_kernel[input_channel][output_channel], True)

                    d_loss_over_padding_output_matrix_col_index += 1

                d_loss_over_padding_output_matrix_row_index += 1

            d_loss_over_padding_output.append(d_loss_over_padding_output_matrix)

        # stores the gradient matrices of loss with respect to padding input, that is input of this layer
        d_loss_over_padding_input = []
        # calculate gradient matrix by channel
        for channel in range(self.input_channel_num):
            # stores gradient matrix for this channel
            d_loss_over_padding_input_matrix = np.zeros(self.input_size, dtype=np.float64)
            # calculate gradient value by element, that is equal to the value of element on corresponding location
            for row in range(self.input_size[0]):
                for col in range(self.input_size[1]):
                    d_loss_over_padding_input_matrix[row, col] = d_loss_over_padding_output[channel][row+self.padding_size[0], col+self.padding_size[1]]
            d_loss_over_padding_input.append(d_loss_over_padding_input_matrix)

        return d_loss_over_pooling_input, d_loss_over_conv, d_loss_over_kernel, d_loss_over_bias, d_loss_over_padding_output, d_loss_over_padding_input

    def forward_calculate(self):
        # padding_result = self.padding()
        convolution_result = self.convolution()
        activation_result = self.conv_activate(convolution_result)
        pooling_result, _ = self.pooling(activation_result)

        return pooling_result


class FullyConnectedLayer:
    def __init__(self, input_im, f_input_channel_num, f_weight_matrix, f_bias_vector, f_activation_func, *f_before_output):
        self.input_im = input_im
        self.input_channel_num = f_input_channel_num
        self.weight_matrix = f_weight_matrix
        self.bias_vector = f_bias_vector
        self.activation_func = f_activation_func
        self.weight_before_output = 0
        self.bias_before_output = 0

        if len(f_before_output):
            self.weight_before_output = f_before_output[0]
            self.bias_before_output = f_before_output[1]

    def flatten(self):
        # get height and width of input image
        h, w = self.input_im[0].shape[0], self.input_im[0].shape[1]
        # a column vector storing flatten result
        flatten_result = np.zeros((h*w*self.input_channel_num, 1), dtype=np.float64)
        # flatten input image matrices by channel
        for channel in range(self.input_channel_num):
            flattened_input = np.resize(self.input_im[channel], (h*w, 1))
            for f_i in range(h*w*channel, h*w*(channel+1)):
                flatten_result[f_i, 0] = flattened_input[f_i-h*w*channel, 0]

        return flatten_result

    def linear_transformation(self, input_vector):
        """
        :param input_vector: a column vector
        :return: a column vector after linear transformation
        """

        output_vector = np.dot(self.weight_matrix, input_vector) + self.bias_vector
        return output_vector

    def nonlinear_transformation(self, input_vector):
        """
        :param input_vector: a column vector
        :return: a column vector after activation
        """
        output_vector = activation_function(self.activation_func, input_vector)
        return output_vector

    def linear_to_output(self, input_vector):
        """
        :param input_vector: a column vector
        # :param weight_before_output: weight matrix
        # :param bias_before_output: bias vector
        :return: output vector
        """
        output_vector = np.dot(self.weight_before_output, input_vector) + self.bias_before_output
        # self.weight_before_output = weight_before_output
        # self.bias_before_output = bias_before_output

        return output_vector

    def fc_grad_af(self, af_input_vector, af_output_vector, *grad_af_value):
        """
        :param af_input_vector: the input vector of activation function
        :param af_output_vector: the output vector of activation function
        :param grad_af_value: if self.activation_func is not a str, then it needs to calculate the gradient in advance
        :return: the gradient of the activation function
        """

        # get the row number of weight matrix, which is also the row number and the column number of gradient matrix of activation function
        row_num = self.weight_matrix.shape[0]
        # the gradient matrix
        result_matrix = np.zeros((row_num, row_num), dtype=np.float64)

        # if self.activation_func is a str, which means the kind of activation function is known, then calculate its gradient
        # otherwise, the gradient value should be calculated in advance
        if isinstance(self.activation_func, str):
            if self.activation_func == 'relu':
                for row in range(row_num):
                    result_matrix[row, row] = af_output_vector[row, 0]
                return result_matrix

            if self.activation_func == 'sigmoid':
                for row in range(row_num):
                    result_matrix[row, row] = af_output_vector[row, 0] * (1-af_output_vector[row, 0])
                return result_matrix

            if self.activation_func == 'tanh':
                for row in range(row_num):
                    result_matrix[row, row] = 1 - np.power(af_output_vector[row, 0], 2)
                return result_matrix

            if self.activation_func == 'softmax':
                for row in range(row_num):
                    for col in range(row_num):
                        if row == col:
                            result_matrix[row, col] = (1-af_output_vector[row, 0])*af_input_vector[row, 0]/af_output_vector[row, 0]
                        else:
                            result_matrix[row, col] = -af_output_vector[row, 0]*af_output_vector[col, 0]
                return result_matrix

        else:
            return grad_af_value[0]

    def fc_gradient_backpropagation(self, d_loss_over_output, activation_result, linear_result, input_vector, *grad_af_value):
        """
        :param d_loss_over_output: \frac {\partial loss}{\partial output}
        :param activation_result: the result of activation function
        :param linear_result: the result of linear transformation
        :param input_vector: the input vector of linear transformation, that is the result of flatten layer
        :param grad_af_value: the gradient matrix of activation function
        :return: \frac {\partial loss}{\partial weight_before_output}, \frac {\partial output}{\partial bias_before_output},
                 \frac {\partial loss}{\partial af_result}, \frac {\partial loss}{\partial z}, \frac {\partial loss}{\partial linear_weight},
                 \frac {\partial loss}{\partial linear_bias}, \frac {\partial loss}{\partial linear_input}(that is \frac {\partial loss}{\partial flatten_result})
        """
        if isinstance(self.weight_before_output, int) == False:
            # resize d_loss_over_output to a column vector
            d_loss_over_output = np.resize(d_loss_over_output, (len(d_loss_over_output), 1))

            # calculate the gradient of loss with respect to weight_before_output
            d_loss_over_weight_before_output = np.dot(d_loss_over_output, np.resize(activation_result, (1, len(activation_result))))

            # calculate the gradient of loss with respect to bias_before_output
            d_loss_over_bias_before_output = np.resize(d_loss_over_output, (len(d_loss_over_output), 1))

            # calculate the gradient of loss with respect to activation result
            d_loss_over_activation_result = np.dot(np.transpose(self.weight_before_output), d_loss_over_output)
        else:
            d_loss_over_weight_before_output, d_loss_over_bias_before_output = 0, 0
            # calculate the gradient of loss with respect to activation result
            d_loss_over_activation_result = copy.deepcopy(d_loss_over_output)

        # calculate the gradient of loss with respect to result of linear transformation
        if len(grad_af_value):
            d_loss_over_linear_result = np.dot(np.transpose(self.fc_grad_af(linear_result, activation_result, grad_af_value)), d_loss_over_activation_result)
        else:
            d_loss_over_linear_result = np.dot(np.transpose(self.fc_grad_af(linear_result, activation_result)), d_loss_over_activation_result)

        # calculate the gradient of loss with respect to weight matrix used in linear transformation
        d_loss_over_linear_weight = np.dot(d_loss_over_linear_result, np.transpose(np.resize(input_vector, (len(input_vector), 1))))

        # calculate the gradient of loss with respect to bias vector used in linear transformation
        d_loss_over_linear_bias = copy.deepcopy(d_loss_over_linear_result)

        # calculate the gradient of loss with respect to input of linear transformation, which is the same as the result of flatten layer
        d_loss_over_linear_input = np.dot(np.transpose(self.weight_matrix), d_loss_over_linear_result)

        return d_loss_over_weight_before_output, d_loss_over_bias_before_output, d_loss_over_activation_result, d_loss_over_linear_weight, d_loss_over_linear_bias, d_loss_over_linear_input

    def forward_calculate(self):
        if isinstance(self.input_im, list):
            linear_input = self.flatten()
        else:
            linear_input = self.input_im
        linear_result = self.linear_transformation(linear_input)
        activation_result = self.nonlinear_transformation(linear_result)

        return activation_result

    def forward_calculate_to_output(self):
        activation_result = self.forward_calculate()
        output = self.linear_to_output(activation_result)

        return output


class MyModel(ConvolutionLayer, FullyConnectedLayer):
    def __init__(self, input_im, f_input_channel_num, f_padding_size, f_conv_kernel, f_conv_kernel_size, f_conv_bias, f_conv_stride_size, f_conv_af, f_pooling_type, f_pooling_kernel_size, f_pooling_stride_size, f_linear_weight, f_linear_bias, f_fc_af, f_weight_before_output, f_bias_before_output, f_initial_lr):
        ConvolutionLayer.__init__(self, input_im, f_input_channel_num, f_padding_size, f_conv_kernel_size, f_conv_kernel, f_conv_bias, f_conv_stride_size, f_conv_af, f_pooling_type, f_pooling_kernel_size, f_pooling_stride_size)
        self.conv1 = ConvolutionLayer(input_im, f_input_channel_num, f_padding_size, f_conv_kernel_size, f_conv_kernel, f_conv_bias, f_conv_stride_size, f_conv_af, f_pooling_type, f_pooling_kernel_size, f_pooling_stride_size)
        FullyConnectedLayer.__init__(self, self.conv1.forward_calculate(), self.conv1.output_channel_num, f_linear_weight[0], f_linear_bias[0], f_fc_af[0])
        self.fc1 = FullyConnectedLayer(self.conv1.forward_calculate(), self.conv1.output_channel_num, f_linear_weight[0], f_linear_bias[0], f_fc_af[0])
        FullyConnectedLayer.__init__(self, self.fc1.forward_calculate(), 1, f_linear_weight[1], f_linear_bias[1], f_fc_af[1], f_weight_before_output, f_bias_before_output)
        self.fc2 = FullyConnectedLayer(self.fc1.forward_calculate(), 1, f_linear_weight[1], f_linear_bias[1], f_fc_af[1], f_weight_before_output, f_bias_before_output)

        self.initial_learning_rate = f_initial_lr
        self.lr = self.initial_learning_rate


    def predict(self):
        return activation_function('softmax', self.fc2.forward_calculate_to_output())

    def cross_entropy_loss(self, f_class_num, true_value, f_lambda_vector):
        """
        :param f_class_num: a column vector storing the number of classes
        :param true_value: a column vector storing the true result of classification
        :param f_lambda_vector: a list storing lambda weights before the L2 norm of the parameters
        :return: the value of cross entropy loss with l2 regularization
        """
        y_predictor = activation_function('softmax', self.predict())
        f_loss = 0
        for f_i in range(f_class_num):
            f_loss -= np.log(y_predictor[f_i, 0]) * true_value[f_i, 0]

        norm_conv_weight = 0
        norm_conv_bias = 0
        for output_channel in range(self.conv1.output_channel_num):
            norm_conv_bias += np.power(np.linalg.norm(self.conv1.conv_bias[output_channel]), 2)
            for input_channel in range(self.conv1.input_channel_num):
                norm_conv_weight += np.power(np.linalg.norm(self.conv1.conv_kernel[output_channel][input_channel]), 2)

        l2_regularization_conv1_weight = f_lambda_vector[0] * norm_conv_weight
        l2_regularization_conv1_bias = f_lambda_vector[1] * norm_conv_bias

        l2_regularization_fc1_weight = f_lambda_vector[2] * np.power(np.linalg.norm(self.fc1.weight_matrix), 2)
        l2_regularization_fc1_bias = f_lambda_vector[3] * np.power(np.linalg.norm(self.fc1.bias_vector), 2)
        l2_regularization_fc2_weight = f_lambda_vector[4] * np.power(np.linalg.norm(self.fc2.weight_matrix), 2)
        l2_regularization_fc2_bias = f_lambda_vector[5] * np.power(np.linalg.norm(self.fc2.bias_vector), 2)
        l2_regularization_fc2_weight_before_output = f_lambda_vector[6] * np.power(np.linalg.norm(self.fc2.weight_before_output), 2)
        l2_regularization_fc2_bias_before_output = f_lambda_vector[7] * np.power(np.linalg.norm(self.fc2.bias_before_output), 2)

        f_loss += (l2_regularization_conv1_weight + l2_regularization_conv1_bias + l2_regularization_fc1_weight + l2_regularization_fc1_bias + l2_regularization_fc2_weight + l2_regularization_fc2_bias + l2_regularization_fc2_weight_before_output + l2_regularization_fc2_bias_before_output)

        return f_loss

    def gradient_cross_entropy_loss_over_output(self, f_class_num, true_value):
        """
        :param f_class_num: a column vector storing the number of classes
        :param true_value: a column vector storing the true result of classification
        :return: \frac {\partial loss}{\partial output}, loss of which is cross entropy loss
        """
        d_loss_over_output = np.zeros((f_class_num, 1), dtype=np.float64)
        y_predictor = activation_function('softmax', self.predict())
        for f_i in range(f_class_num):
            d_loss_over_output[f_i, 0] -= true_value[f_i, 0] / y_predictor[f_i, 0]

        return d_loss_over_output

    def mse_loss(self, f_class_num, true_value, f_lambda_vector):
        """
        :param f_class_num: a column vector storing the number of classes
        :param true_value: a column vector storing the true result of classification
        :param f_lambda_vector: a list storing lambda weights before the L2 norm of the parameters
        :return: the value of cross entropy loss with l2 regularization
        """
        y_predictor = activation_function('softmax', self.predict())
        f_loss = 0
        for f_i in range(f_class_num):
            f_loss += np.power(y_predictor[f_i, 0]-true_value[f_i, 0], 2)

        f_loss /= f_class_num

        norm_conv_weight = 0
        norm_conv_bias = 0
        for output_channel in range(self.conv1.output_channel_num):
            norm_conv_bias += np.power(np.linalg.norm(self.conv1.conv_bias[output_channel]), 2)
            for input_channel in range(self.conv1.input_channel_num):
                norm_conv_weight += np.power(np.linalg.norm(self.conv1.conv_kernel[output_channel][input_channel]), 2)

        l2_regularization_conv1_weight = f_lambda_vector[0] * norm_conv_weight
        l2_regularization_conv1_bias = f_lambda_vector[1] * norm_conv_bias

        l2_regularization_fc1_weight = f_lambda_vector[2] * np.power(np.linalg.norm(self.fc1.weight_matrix), 2)
        l2_regularization_fc1_bias = f_lambda_vector[3] * np.power(np.linalg.norm(self.fc1.bias_vector), 2)
        l2_regularization_fc2_weight = f_lambda_vector[4] * np.power(np.linalg.norm(self.fc2.weight_matrix), 2)
        l2_regularization_fc2_bias = f_lambda_vector[5] * np.power(np.linalg.norm(self.fc2.bias_vector), 2)
        l2_regularization_fc2_weight_before_output = f_lambda_vector[6] * np.power(
            np.linalg.norm(self.fc2.weight_before_output), 2)
        l2_regularization_fc2_bias_before_output = f_lambda_vector[7] * np.power(
            np.linalg.norm(self.fc2.bias_before_output), 2)

        f_loss += (l2_regularization_conv1_weight + l2_regularization_conv1_bias + l2_regularization_fc1_weight + l2_regularization_fc1_bias + l2_regularization_fc2_weight + l2_regularization_fc2_bias + l2_regularization_fc2_weight_before_output + l2_regularization_fc2_bias_before_output)

        return f_loss

    def gradient_mse_loss_over_output(self, f_class_num, true_value):
        """
        :param f_class_num: a column vector storing the number of classes
        :param true_value: a column vector storing the true result of classification
        :return: \frac {\partial loss}{\partial output}, loss of which is cross entropy loss
        """
        y_predictor = activation_function('softmax', self.predict())
        d_loss_over_output = 2/f_class_num * (y_predictor-true_value)

        return d_loss_over_output

    def gradient_backpropagation_between_layers(self, d_loss_over_output, *grad_af_value):
        """
        :param d_loss_over_output: \frac {\partial loss}{\partial output}
        :param grad_af_value: the gradient value of activation function calculated in advance
        :return: the gradient between Conv1 and FC1 \frac {\partial loss}{\partial conv_pooling_result} and
                 the gradient between FC1 and FC2 \frac {\partial loss}{\partial fc1_result}
        """
        # get the output channel number, size of output matrix for each channel of conv1
        conv_output_channel_num = self.conv1.output_channel_num
        f_conv_output_size = self.conv1.forward_calculate()[0].shape

        # calculate forward result of flatten, linear and activation of fc1
        fc1_flatten_result = self.fc1.flatten()
        fc1_linear_result = self.fc1.linear_transformation(fc1_flatten_result)
        fc1_activation_result = self.fc1.nonlinear_transformation(fc1_linear_result)
        # calculate forward result of linear and activation of fc2
        fc2_linear_result = self.fc2.linear_transformation(fc1_activation_result)
        fc2_activation_result = self.fc2.nonlinear_transformation(fc2_linear_result)
        # get all the gradients in fc1
        if len(grad_af_value) == 3 and grad_af_value[2] != 'none':
            f_d_loss_over_fc1_output = self.fc2.fc_gradient_backpropagation(d_loss_over_output, fc2_activation_result, fc2_linear_result, fc1_activation_result, grad_af_value[2])[-1]
        else:
            f_d_loss_over_fc1_output = self.fc2.fc_gradient_backpropagation(d_loss_over_output, fc2_activation_result, fc2_linear_result, fc1_activation_result)[-1]
        if len(grad_af_value) >= 2 and grad_af_value[1] != 'none':
            fc1_gradient_bp = self.fc1.fc_gradient_backpropagation(f_d_loss_over_fc1_output, fc1_activation_result, fc1_linear_result, fc1_flatten_result, grad_af_value[1])
        else:
            fc1_gradient_bp = self.fc1.fc_gradient_backpropagation(f_d_loss_over_fc1_output, fc1_activation_result, fc1_linear_result, fc1_flatten_result)

        # get \frac {\partial loss}{\partial fc1_result}(that is \frac {\partial loss}{\partial fc2_linear_input}) and \frac {\partial loss}{\partial fc1_flatten_result}
        d_loss_over_fc1_result, d_loss_over_fc1_flatten_result = fc1_gradient_bp[2], fc1_gradient_bp[-1]

        # stores \frac {\partial loss}{\partial conv1_result}
        d_loss_over_conv_result = []
        # calculate gradient by channel
        for channel in range(conv_output_channel_num):
            # stores gradient for this channel
            d_loss_over_conv_result_matrix = np.zeros(f_conv_output_size, dtype=np.float64)
            # gain gradient by element
            for row in range(f_conv_output_size[0]):
                for col in range(f_conv_output_size[1]):
                    d_loss_over_conv_result_matrix[row, col] = d_loss_over_fc1_flatten_result[row*f_conv_output_size[1] + col, 0]
            d_loss_over_conv_result.append(d_loss_over_conv_result_matrix)

        return d_loss_over_conv_result, d_loss_over_fc1_result

    def gradient_descent(self, f_epoch, f_decay_rate, f_d_loss_over_output):
        """
        to renew all the eight parameters used in conv1, fc1 and fc2
        :param f_epoch: the number of iteration about batch
        :param f_decay_rate: the decay rate of learning rate with respect to time, that is epoch number
        :param f_d_loss_over_output: \frac {\partial loss}{\partial output}
        # :param f_class_num: the number of all classes
        # :param true_value: the true value for a certain x from training dataset
        """
        self.lr = self.lr * np.power(f_decay_rate, f_epoch)
        f_activation_result = self.conv1.conv_activate(self.conv1.convolution())
        f_padding_list = self.conv1.padding()
        _, f_pooling_gradient = self.conv1.pooling(f_activation_result)
        # f_d_loss_over_output = self.gradient_loss_over_output(f_class_num, true_value)
        f_d_loss_over_pooling_output, f_d_loss_over_fc1_result = self.gradient_backpropagation_between_layers(f_d_loss_over_output)
        _, _, f_d_loss_over_conv_kernel, f_d_loss_over_conv_bias, _, _ = self.conv1.conv_gradient_backpropagation(f_d_loss_over_pooling_output, f_pooling_gradient, f_activation_result, f_padding_list)

        self.conv1.conv_kernel = [[self.conv1.conv_kernel[f_i][f_j]-self.lr*f_d_loss_over_conv_kernel[f_i][f_j] for f_j in range(self.conv1.input_channel_num)] for f_i in range(self.conv1.output_channel_num)]
        self.conv1.conv_bias = [self.conv1.conv_bias[f_i]-self.lr*f_d_loss_over_conv_bias[f_i] for f_i in range(self.conv1.output_channel_num)]

        f_fc1_input_vector = x.fc1.flatten()
        f_fc1_linear_result = self.fc1.linear_transformation(f_fc1_input_vector)
        f_fc1_activation_result = self.fc1.nonlinear_transformation(f_fc1_linear_result)
        _, _, _, f_d_loss_over_fc1_linear_weight, f_d_loss_over_fc1_linear_bias, _ = self.fc1.fc_gradient_backpropagation(f_d_loss_over_fc1_result, f_fc1_activation_result, f_fc1_linear_result, f_fc1_input_vector)

        self.fc1.weight_matrix -= self.lr * f_d_loss_over_fc1_linear_weight
        self.fc1.bias_vector -= self.lr * f_d_loss_over_fc1_linear_bias

        f_fc2_input_vector = self.fc1.forward_calculate()
        f_fc2_linear_result = self.fc2.linear_transformation(f_fc2_input_vector)
        f_fc2_activation_result = self.fc2.nonlinear_transformation(f_fc2_linear_result)
        f_d_loss_over_weight_before_output, f_d_loss_over_bias_before_output, _, f_d_loss_over_fc2_linear_weight, f_d_loss_over_fc2_linear_bias, _ = self.fc2.fc_gradient_backpropagation(f_d_loss_over_output, f_fc2_activation_result, f_fc2_linear_result, f_fc2_input_vector)

        self.fc2.weight_before_output -= self.lr * f_d_loss_over_weight_before_output
        self.fc2.bias_before_output -= self.lr * f_d_loss_over_bias_before_output
        self.fc2.weight_matrix -= self.lr * f_d_loss_over_fc2_linear_weight
        self.fc2.bias_vector -= self.lr * f_d_loss_over_fc2_linear_bias

        # print([[(np.max(self.conv1.conv_kernel[f_i][f_j]), np.min(self.conv1.conv_kernel[f_i][f_j])) for f_j in range(self.conv1.input_channel_num)] for f_i in range(self.conv1.output_channel_num)])




import mnist_reader
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

train_num = X_train.shape[0]
test_num = X_test.shape[0]
class_num = 10
# print(train_num, test_num)       # 60000 10000
# print(X_test[0][100])

np.random.seed(12345)

input_im_size = (28, 28)
input_channel_num = 1
padding_size = (1, 1)
conv_kernel_size = (3, 3)
output_channel_num = 4
conv_kernel = [[np.random.uniform(-0.01, 0.01, conv_kernel_size) for _ in range(input_channel_num)] for _ in range(output_channel_num)]
conv_stride_size = (1, 1)
conv_bias_size = (int((input_im_size[0]+2*padding_size[0]-conv_kernel_size[0])/conv_stride_size[0]+1), int((input_im_size[1]+2*padding_size[1]-conv_kernel_size[1])/conv_stride_size[1]+1))
conv_bias = [np.random.uniform(-0.01, 0.01, conv_bias_size) for _ in range(output_channel_num)]

conv_af = 'relu'
pooling_type = 'max'
pooling_kernel_size = (2, 2)
pooling_stride_size = (2, 2)
conv_output_size = (np.ceil((conv_bias_size[0]-pooling_kernel_size[0])/pooling_stride_size[0]+1), np.ceil((conv_bias_size[1]-pooling_kernel_size[1])/pooling_stride_size[1]+1))
hidden_vector_size_1 = 1000
hidden_vector_size_2 = 100
linear_weight = []
linear_bias = []
linear_weight.append(np.random.normal(0, 0.01, (hidden_vector_size_1, int(conv_output_size[0]*conv_output_size[1]*output_channel_num))))
linear_bias.append(np.random.normal(0, 0.01, (hidden_vector_size_1, 1)))
linear_weight.append(np.random.normal(0, 0.01, (hidden_vector_size_2, hidden_vector_size_1)))
linear_bias.append(np.random.normal(0, 0.01, (hidden_vector_size_2, 1)))
fc_af = ['relu', 'relu']
weight_before_output = np.random.normal(0, 0.01, (class_num, hidden_vector_size_2))
bias_before_output = np.random.normal(0, 0.01, (class_num, 1))
initial_lr = np.power(10, -3.)
lambda_vector = np.array([1, 1, 1, 1, 1, 1, 1, 1]) * np.power(10, -3.)
decay_rate = 0.8


loss = []
accuracy = []
num_epochs = 1000
batch_size = 50

# x = MyModel(turn_to_matrix(X_train[0], 28, 28), input_channel_num, padding_size, conv_kernel, conv_kernel_size,
#             conv_bias, conv_stride_size, conv_af, pooling_type, pooling_kernel_size, pooling_stride_size, linear_weight, linear_bias,
#             fc_af, weight_before_output, bias_before_output, initial_lr)
# y_predict = x.predict()
# y_true = np.zeros((10, 1), dtype=np.int8)
# y_true[y_train[0], 0] = 1
# x_loss = x.cross_entropy_loss(class_num, y_true, lambda_vector)
# x.gradient_descent(1, decay_rate, class_num, y_true)
# print(x_loss)

for epoch in range(num_epochs):
    random_index = np.random.randint(low=0, high=train_num, size=batch_size)
    indicator = 1
    for batch in random_index:
        x = MyModel(turn_to_matrix(X_train[batch], 28, 28), input_channel_num, padding_size, conv_kernel, conv_kernel_size,
                    conv_bias, conv_stride_size, conv_af, pooling_type, pooling_kernel_size, pooling_stride_size, linear_weight, linear_bias,
                    fc_af, weight_before_output, bias_before_output, initial_lr)
        y_predict = x.predict()
        y_true = np.zeros((10, 1), dtype=np.int8)
        y_true[y_train[batch], 0] = 1

        x_loss = x.cross_entropy_loss(class_num, y_true, lambda_vector)
        loss.append(x_loss)
        d_loss_over_yhat = x.gradient_cross_entropy_loss_over_output(class_num, y_true)
        # x_loss = x.mse_loss(class_num, y_true, lambda_vector)
        # loss.append(x_loss)
        # d_loss_over_yhat = x.gradient_mse_loss_over_output(class_num, y_true)

        x.gradient_descent(epoch*batch+1, decay_rate, d_loss_over_yhat)

        # print(y_predict, indicator)
        indicator += 1
    # print(loss)

    epoch_accuracy = 0
    for test_item in range(test_num):
        x = MyModel(turn_to_matrix(X_test[test_item], 28, 28), input_channel_num, padding_size, conv_kernel, conv_kernel_size,
                    conv_bias, conv_stride_size, conv_af, pooling_type, pooling_kernel_size, pooling_stride_size, linear_weight, linear_bias,
                    fc_af, weight_before_output, bias_before_output, initial_lr)
        y_predict = x.predict()
        epoch_accuracy += int(np.where(y_predict == np.max(y_predict))[0][0] == y_test[test_item])

        # if (test_item+1) % 100 == 0:
        #     print(test_item)

    accuracy.append(epoch_accuracy / test_num)
    # print(accuracy)

    if (epoch+1) % 200 == 0:
        # print(loss)
        # print(accuracy)

        fig, ax1 = plt.subplots()
        ax1.plot(range(epoch), loss, color='blue', label='Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')

        ax2 = ax1.twinx()
        ax2.plot(range(epoch), accuracy, color='red', label='Accuracy')
        ax2.set_ylabel('Accuracy')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.title(f'Loss and Accuracy for epoch {epoch+1}')
        plt.savefig(f'figure//loss_and_accuracy_for_epoch{epoch+1}.png', dpi=300)
        plt.show()

        # plt.figure()
        # x_axis = list(list(range(0, epoch+1, 1000)))
        # plt.plot(x_axis, [loss[i] for i in x_axis], marker='o', label='loss')
        # plt.draw()

        # plt.figure()
        # plt.plot(x_axis, [accuracy[i] for i in x_axis], marker='o', label='accuracy')
        # plt.draw()

        # plt.show()

        # user_input = input("continue？(y/n): ")
        # if user_input.lower() != 'y':
        #     break

# plt.show()



# class MyModel:
#     def __init__(self, input_im, input_channel_num, conv_activation_func, pooling_type, weight_matrix, fc_activation_func):
#         self.input_im = input_im
#         self.input_channel_num = input_channel_num
#         self.conv_activation_func = conv_activation_func
#         self.pooling_type = pooling_type
#         self.weight_matrix = weight_matrix
#         self.fc_activation_func = fc_activation_func
#
#     def construct_model(self, padding_size, conv_stride_size, conv_kernel_info, conv_b, pooling_stride_size, pooling_kernel_size, fc_b):
#         conv_1 = ConvolutionLayer(self.input_im, self.input_channel_num, self.conv_activation_func, self.pooling_type)
#         # padding_im = conv_1.padding(padding_size)
#         conv_result = conv_1.conv(conv_stride_size, padding_size, conv_b, conv_kernel_info)
#         conv_activation_result = conv_1.activate(conv_result)
#         conv_output_channel_num = len(conv_result)
#         conv_pooling_result = conv_1.pooling(pooling_kernel_size, pooling_stride_size, conv_activation_result, conv_output_channel_num)
#
#         fc_1 = FullyConnectedLayer(conv_pooling_result, conv_output_channel_num, self.weight_matrix[0], fc_b[0], self.fc_activation_func[0])


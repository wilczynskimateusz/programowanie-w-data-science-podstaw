# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 23:26:31 2020

@author: blood
"""

import operator as op
from copy import deepcopy

class ShapeMismatchError(Exception):
     def __init__(self, message):
            super().__init__(message)
            
class Matrix:
    def __init__(self, list_of_values, nrows=1, ncols=1):
        if(nrows * ncols) != len(list_of_values):
            raise ShapeMismatchError(
                'Unable to fully fill out the rows and columns with provided list of values'
            )

        rows = []
        for i in range(0, nrows):
            rows.append(list_of_values[i*nrows:(i+1)*ncols])
        self.rows = rows
        
        cols = []
        for i in range(0, ncols):
            col = []
            for j in range(0, nrows):
                col.append(self.rows[j][i])
            cols.append(col)
            
        self.cols = cols
        self.shape = (len(rows), len(cols))

    def __getitem__(self, rownum):
        return self.rows[rownum]
    
    def __delitem__(self, rownum):
        del self.rows[rownum]
    
    def __validate_shape(self, matrix):
        if self.shape != matrix.shape:
            raise ShapeMismatchError(
                'The given matrix has different shape than the origin matrix')

    def __basic_operation(self, matrix, operation):
        new_list = []
        for row in range(0, len(self.rows)):
            new_row = [operation(x,y) for x,y in zip(self.rows[row], matrix.rows[row])]
            for nr in new_row:
                new_list.append(nr)

        return new_list
    
    def add(self, matrix, inplace=False):
        self.__validate_shape(matrix)
        new_list = self.__basic_operation(matrix, op.add)
            
        if(inplace == True):
            self = Matrix(new_list, len(self.rows), len(self.cols))

        return (self if inplace == True else Matrix(new_list, len(self.rows), len(self.cols)))
    
    def subtract(self, matrix, inplace=False):
        self.__validate_shape(matrix)
        new_list = self.__basic_operation(matrix, op.sub)

        if(inplace == True):
            self = Matrix(new_list, len(self.rows), len(self.cols))

        return (self if inplace == True else Matrix(new_list, len(self.rows), len(self.cols)))

    def multiply(self, matrix, inplace=False):
        if(self.shape[1] != matrix.shape[0]):
            raise ShapeMismatchError('Unable to multiply those two matrices')

        new_list = []
        for row in range(0, len(self.rows)):
            for i in range(0, len(self.cols)):
                new_element = 0
                for col in range(0, len(self.cols)):
                    new_element += self.rows[row][col] * \
                        matrix.cols[i][col]

                new_list.append(new_element)

        if (inplace == True):
            self = Matrix(new_list, len(self.rows), len(matrix.cols))
            
        return (self if inplace == True else Matrix(new_list, len(self.rows), len(matrix.cols)))
    
    def multiply_by_number(self, number=1, inplace=False):
        new_list = []
                
        for row in range(0, len(self.rows)):
            for col in range(0, len(self.cols)):
                new_list.append(self[row][col] * number)
                
        if(inplace == True):
            self = Matrix(new_list, len(self.rows), len(self.cols))

        return (self if inplace == True else Matrix(new_list, len(self.rows), len(self.cols)))
    
    def transpose(self, inplace=False):
        if inplace == True:
            rows = deepcopy(self.rows)
            cols = (self.cols)

            self.rows = cols
            self.cols = rows
        
        new_list = []
        for col in self.cols:
            for element in col:
                new_list.append(element)
        
        return (self if inplace == True else Matrix(new_list, len(self.cols), len(self.rows)))

    def det(self):
        if len(self.rows) != len(self.cols):
            raise ShapeMismatchError(
                'Cannot compute the determinant of a non-square matrix'
            )

        if len(self.rows) == 1:
            det = self.rows[0][0]

        if len(self.rows) == 2:
            det = self.rows[0][0] * self.rows[1][1] - self.rows[0][1] * self.rows[1][0]

        if len(self.rows) == 3:
            helper_matrix = deepcopy(self)
            helper_matrix.rows.append(helper_matrix.rows[0])
            helper_matrix.rows.append(helper_matrix.rows[1])
            
            det = 0
            for row in range(0, len(self.rows)):
                result = 1
                counter = 0
                for col in range(0, len(self.cols)):
                    result *= helper_matrix[row+counter][col]
                    counter += 1

                det += result
            for row in range(0, len(self.rows)):
                result = 1
                counter = 0
                for col in range(-len(self.cols), 0):
                    result *= helper_matrix[row+counter][col]
                    counter += 1

                det -= result
                
        return det

    def complement(self):
        if len(self.rows) != len(self.cols):
            raise ShapeMismatchError(
               'Cannot compute complementary matrix for non-square matrix'
            )
        det_list = []
        det_matrix = []
        for i in range(0, len(self.rows)):
            for col in range(0, len(self.cols)):
                helper_matrix = deepcopy(self)
                for row in range(0, len(self.rows)):
                    del helper_matrix[row][col]

                del helper_matrix[i]
                
                helper_list = []
                for row in helper_matrix.rows:
                    for element in row:
                        helper_list.append(element)
                det_list.append(helper_list)
            
        for matrix in det_list:
            det_matrix.append(Matrix(matrix, len(self.rows)-1, len(self.cols)-1).det())
                
        for i in range(1, len(det_matrix), 2):
            value = det_matrix[i]
            det_matrix[i] = -value
            
        return Matrix(det_matrix, len(self.rows), len(self.cols))
    
    def inverse(self, inplace=False):
        return self.complement().transpose().multiply_by_number(1/self.det())
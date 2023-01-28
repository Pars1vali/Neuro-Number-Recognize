import org.ejml.data.*;
import org.ejml.dense.row.DMatrixVisualization;
import org.ejml.dense.row.RandomMatrices_DDRM;
import org.ejml.ops.DConvertMatrixStruct;
import org.ejml.ops.MatrixIO;
import org.ejml.simple.SimpleMatrix;

import java.io.*;
import java.sql.SQLOutput;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.ejml.dense.row.mult.MatrixVectorMult_FDRM.mult;

public class Main {
    public Main() {
    }

    public static void main(String[] args) throws IOException {
        MnistMatrix[] mnistMatrix = new MnistDataReader().readData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
        MnistMatrix[] mnistMatrix_Test = new MnistDataReader().readData("data/t10k-images.idx3-ubyte","data/t10k-labels.idx1-ubyte");

        Random random = new Random();
        SimpleMatrix weight_0_1 = SimpleMatrix.random_DDRM(784,100,-0.5,0.5,random);
        SimpleMatrix weight_1_2 = SimpleMatrix.random_DDRM(100, 10, -0.5,0.5, random);


        int iterations = 350;
        int images = 1000;
        int images_test = 1000;
        double alpha = 0.005 ;

        for (int i = 0; i < iterations; i++) {
            double error = 0;
            double correct_cnt = 0;

            for (int j = 0; j < images; j++) {

                SimpleMatrix labels = new SimpleMatrix(1,10);
                labels.set(mnistMatrix[j].getLabel(),1);

                SimpleMatrix dropout_matrix = createDropoutMatrix();

                SimpleMatrix layer_0 = layer_0_Create(mnistMatrix[j]);
                SimpleMatrix layer_1 = layer_0.mult(weight_0_1);
                for (int element = 0; element < layer_1.getNumElements(); element++) {
                    if(layer_1.get(element)<0){
                        layer_1.set(element,0.0);
                    }
                    layer_1.set(element,layer_1.get(element)*dropout_matrix.get(element));
                }

                SimpleMatrix layer_2 = layer_1.mult(weight_1_2);
                error += Math.pow((labels.minus(layer_2)).elementSum(),2);

                double max_elem = layer_2.get(0);
                int index_max_elem = 0;
                for (int index = 1; index < layer_2.getNumElements(); index++) {
                    if (max_elem < layer_2.get(index)) {
                        max_elem = layer_2.get(index);
                        index_max_elem = index;
                    }
                }
                correct_cnt += (index_max_elem==mnistMatrix[j].getLabel())?1.0:0.0;

                SimpleMatrix layer_2_delta = labels.minus(layer_2);
                SimpleMatrix layer_1_delta = layer_2_delta.mult(weight_1_2.transpose());

                for (int number_elements = 0; number_elements < layer_1_delta.getNumElements(); number_elements++) {
                    if(layer_1.get(number_elements)<0){
                        layer_1_delta.set(number_elements,0.0);
                    }
                }
                //layer_1_delta = layer_1_delta.mult(dropout_matrix);


                weight_0_1 = weight_0_1.plus((layer_0.transpose().mult(layer_1_delta)).scale(alpha));
                weight_1_2 = weight_1_2.plus((layer_1.transpose().mult(layer_2_delta)).scale(alpha));

               // System.out.println("\t\t right - " + mnistMatrix[j].getLabel() + " -> real - " + index_max_elem);


            }
            if(i%10==0){
                double test_error = 0;
                double test_correct_cnt = 0;
                for (int m = 0; m < images_test; m++) {
                    SimpleMatrix labels_t = new SimpleMatrix(1,10);
                    labels_t.set(mnistMatrix_Test[m].getLabel(),1);

                    SimpleMatrix layer_0_t = layer_0_Create(mnistMatrix_Test[m]);
                    SimpleMatrix layer_1_t = layer_0_t.mult(weight_0_1);
                    for (int number_elements = 0; number_elements < layer_1_t.getNumElements(); number_elements++) {
                        if(layer_1_t.get(number_elements)<0){
                            layer_1_t.set(number_elements,0.0);
                        }
                    }

                    SimpleMatrix layer_2_t = layer_1_t.mult(weight_1_2);

                    test_error = Math.pow((labels_t.minus(layer_2_t)).elementSum(),2);

                    double max_elem_t = layer_2_t.get(0);
                    int index_max_elem_t = 0;
                    for (int index = 1; index < layer_2_t.getNumElements(); index++) {
                        if (max_elem_t < layer_2_t.get(index)) {
                            max_elem_t = layer_2_t.get(index);
                            index_max_elem_t = index;
                        }
                    }
                    test_correct_cnt += (index_max_elem_t==mnistMatrix_Test[m].getLabel())?1.0:0.0;
                }
                System.out.println("\n" +
                        "I: " + i +
                        " Test-Err: " + test_error/images_test +
                        " Test-Acc: " + test_correct_cnt/images_test +
                        " Train-Err: " + error/images +
                        " Train-Acc: " + correct_cnt/images);
            }

        }



    }
    private static SimpleMatrix layer_0_Create(final MnistMatrix matrixNumber) {
       SimpleMatrix layer_0 = new SimpleMatrix(1, 784);
        for (int r = 0, i = 0; r < matrixNumber.getNumberOfRows(); r++ ) {
            for (int c = 0; c < matrixNumber.getNumberOfColumns(); c++, i++) {
                layer_0.set(i,matrixNumber.getValue(r,c)/255.0);
            }
        }
        return layer_0;

    }
    private static SimpleMatrix createDropoutMatrix(){
        SimpleMatrix dropout_matrix = new SimpleMatrix(1,100);
        for (int dr_index = 0; dr_index < dropout_matrix.getNumElements(); dr_index++) {
                   dropout_matrix.set(dr_index,(int)(Math.random()*2));
        }

        int count_1 = 0;
        int count_0 = 0;
        for (int z = 0; z < 100; z++) {
            int number = (int) dropout_matrix.get(z);
            if(number==1){
                count_1++;
            }else if(number==0){
                count_0++;
            }
        }

        if(count_0>count_1 ){
            int l = 0;
            while (count_0!=count_1 && l<100){
                if(dropout_matrix.get(l)==0){
                    dropout_matrix.set(l,1);
                    count_0--;
                    count_1++;
                }
                l++;
            }
        } else if (count_0<count_1 ) {
            int l = 0;
            while (count_0!=count_1 && l<100){
                if(dropout_matrix.get(l)==1){
                    dropout_matrix.set(l,0);
                    count_0++;
                    count_1--;
                }
                l++;
            }
        }
        return dropout_matrix;
    }


}


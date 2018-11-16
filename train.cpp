
#include "train.h"
#include "utils.h"

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace ml;


void getAnnHOGFeatures(const Mat image, Mat& features) {
	//
	HOGDescriptor hog(Size(32, 32), Size(16, 16), Size(8, 8), Size(8, 8), 3);
	std::vector<float> descriptor;

	Mat trainImg = Mat(hog.winSize, CV_32S);
	resize(image, trainImg, hog.winSize);

	//����
	hog.compute(trainImg, descriptor, Size(8, 8));

	Mat mat_featrue(descriptor);
	mat_featrue.copyTo(features);
}


//Artificial Neural Networks �˹������� ����֪��
//��ͼ http://docs.opencv.org/3.3.0/dc/dd6/ml_intro.html#ml_intro_ann
//1������Ҫʶ�������ͨ��ĳ��;�����x(��������)������
//2�����������ݷֳ�y(���ز���Ԫ����)������(��Ԫ) �������ز����ظ��������
//3���ѵ�2������� ���������� Ȼ�����z(��������)�����
void doAnnZhTrain() {
	printf("Begin to taining ann ZH...\n");
	//ͼƬ��ַ������Ӧ�ı�ǩ
	vector<TrainStruct> ann_data;
	Mat train_samples;
	//�ļ�������1-31
	for (int i = 0; i < 31; ++i) {
		char path[100];
		sprintf(path, "%s/%d", ANN_CH_SAMPLE, i + 1);
		//ͼƬ��ַ
		vector<string> files;
		getFiles(path, files);
		//���ļ�·���������±��ǩ���뼯��
		for (string file : files) {
			ann_data.push_back({ file,i });
		}
	}

	//�� ����3������ ���Ϊ2 ��0��1������Ϊ��0�࣬��2�������ǵ�1��
	//������������:
	//����3������ ����(������)Ϊ4(2x2)
	//1,1,1,1
	//1,1,1,1
	//1,1,1,1
	//������ǩ����ҪΪ
	//1,0
	//1,0
	//0,1
	Mat samples;
	vector<int> label;
	for (TrainStruct data : ann_data) {
		//ֱ�Ӷ����Ҷ�ͼ
		Mat image = imread(data.file, IMREAD_GRAYSCALE);
		if (image.empty()) {
			printf("Load failed ANN ZH image: %s.\n", data.file.c_str());
			continue;
		}
		//        resize(image, image, Size(ANN_COLS,ANN_ROWS));
		threshold(image, image, 0, 255, THRESH_BINARY + THRESH_OTSU);
		Mat feature;
		getAnnHOGFeatures(image, feature);
		feature = feature.reshape(1, 1);
		//2������ һ������һ����ǩ
		samples.push_back(feature);
		label.push_back(data.label);
	}
	//samples ���汣�������еĴ�ѵ����������������
	samples.convertTo(samples, CV_32FC1);
	//������ǩ
	Mat train_label = Mat::zeros((int)label.size(), 31, CV_32FC1);
	for (int i = 0; i < train_label.rows; ++i) {
		train_label.at<float>(i, label[i]) = 1.f;
	}
	cout << "Tag��" << train_label << endl;

	//ѵ������
	Ptr<TrainData> train_data = TrainData::create(samples, SampleTypes::ROW_SAMPLE, train_label);

	Ptr<ANN_MLP> classifier = ANN_MLP::create();
	Mat layers;
	layers.create(1, 3, CV_32SC1);
	//�����
	layers.at<int>(0) = samples.cols;   //ҪԤ�������(ͼƬ)�Ŀ� 
	layers.at<int>(1) = 62; //���ز����� �������Ҫ������ ���������ǲ��ɼ���
	//�����
	layers.at<int>(2) = 31;//����� 31�� 31������ʡ�� Ҳ����train_label�Ŀ�
	//cout << "��ǩ��" << layers << endl;
	classifier->setLayerSizes(layers);
	//http://docs.opencv.org/3.3.0/d0/dce/classcv_1_1ml_1_1ANN__MLP.html#a16998f97db903c1c652e68f342240524
	//��ʼ��ÿ����Ԫ�ļ����,���������������������������һ�� Ŀǰ��ȫ֧�ֵ�ֻ��SIGMOID_SYM �����������㷨�еĦ��ͦ£�Ĭ������»ᶼ������Ϊ1
	classifier->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1, 1);
	//bp���� һ���������㷨,����������ѵ�����̵���ֵ�㷨���ٶȿ�
	classifier->setTrainMethod(ANN_MLP::TrainingMethods::BACKPROP);
	//�����㷨����ֹ���� ��������(CV_TERMCRIT_ITER)�������Сֵ(CV_TERMCRIT_EPS)��һ����һ���ﵽ��������ֹѵ��
	classifier->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 30000, 0.0001));


	printf("Prepare data finished, begin to training ANN ZH\n");

	classifier->train(train_data);
	classifier->save(ANN_ZH_XML);
	printf("training ANN ZH chars predict model save: %s\n", ANN_ZH_XML);
}



void doAnnTrain() {
	printf("Begin to taining ann chars preditct models...\n");

	vector<TrainStruct> ann_data;
	Mat train_samples;
	for (int i = 0; i < 34; ++i) {
		char path[100];
		sprintf(path, "%s/%d", ANN_SAMPLE, i);
		vector<string> files;
		getFiles(path, files);
		for (string file : files) {
			ann_data.push_back({ file,i });
		}
	}
	Mat samples;
	vector<int> label;
	for (TrainStruct data : ann_data) {
		Mat image = imread(data.file, IMREAD_GRAYSCALE);
		if (image.empty()) {
			printf("load sample failed image: %s.\n", data.file.c_str());
			continue;
		}
		threshold(image, image, 0, 255, THRESH_BINARY + THRESH_OTSU);
		Mat feature;
		getAnnHOGFeatures(image, feature);
		feature = feature.reshape(1, 1);
		samples.push_back(feature);
		label.push_back(data.label);
	}
	samples.convertTo(samples, CV_32FC1);
	Mat train_classes = Mat::zeros((int)label.size(), 34, CV_32FC1);
	for (int i = 0; i < train_classes.rows; ++i) {
		train_classes.at<float>(i, label[i]) = 1.f;
	}

	Ptr<TrainData>  train_data = TrainData::create(samples, SampleTypes::ROW_SAMPLE, train_classes);

	Ptr<ANN_MLP> classifier = ANN_MLP::create();
	Mat layers;
	layers.create(1, 3, CV_32SC1);
	layers.at<int>(0) = samples.cols;
	layers.at<int>(1) = 68;
	layers.at<int>(2) = 34;//�����
	classifier->setLayerSizes(layers);
	classifier->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1, 1);
	classifier->setTrainMethod(cv::ml::ANN_MLP::TrainingMethods::BACKPROP);
	classifier->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 30000, 0.0001));

	printf("Prepare data finished, begin to training ANN\n");


	classifier->train(train_data);
	classifier->save(ANN_XML);
	printf("training ann chars predict model save: %s\n",  ANN_XML);
}









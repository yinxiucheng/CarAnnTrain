


#ifndef Train_hpp
#define  Train_hpp

#include <string>

using namespace std;

#define ANN_ZH_XML "/Users/xiuchengyin/Documents/Tina-NDK/CarAnnTrain/CarAnnTrain/resource/HOG_ANN_ZH_DATA2.xml"
#define ANN_XML "/Users/xiuchengyin/Documents/Tina-NDK/CarAnnTrain/CarAnnTrain/resource/HOG_ANN_DATA2.xml"
#define ANN_CH_SAMPLE "/Users/xiuchengyin/Documents/Tina-NDK/CarAnnTrain/CarAnnTrain/resource/train/ann_zh"
#define ANN_SAMPLE "/Users/xiuchengyin/Documents/Tina-NDK/CarAnnTrain/CarAnnTrain/resource/train/ann"

struct TrainStruct {
	string file;
	int label;
};
void doAnnZhTrain();
void doAnnTrain();

#endif /* Train_hpp */

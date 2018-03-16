#ifndef CAFFE_SOFTMAX_WITH_SPHERE_LOSS_LAYER_HPP_
#define CAFFE_SOFTMAX_WITH_SPHERE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

/**
 * Author: Shi Neng
 * @brief change of softmax loss view layer, use only one kind of class, split the sphere
 * uniformly in 2352
 * instead of only considering the -log(prob) on label class, consider a weighted
 * sum of -log(prob) for classes within a bandwidth of label class.
 */
template <typename Dtype>
  class SoftmaxWithSphereLossLayer : public LossLayer<Dtype> {
    public:
      explicit SoftmaxWithSphereLossLayer(const LayerParameter& param)
        : LossLayer<Dtype>(param),
        softmax_layer_(new SoftmaxLayer<Dtype>(param)) {}
      virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top);
      virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top);

      virtual inline const char* type(){
        return "SoftmaxWithSphereLoss";
      }
      virtual inline int ExactNumBottomBlobs() const { return -1; }
      virtual inline int MinBottomBlobs() const { return 2; }
      virtual inline int MaxBottomBlobs() const { return 3; }
      virtual inline int ExactNumTopBlobs() const { return -1; }
      virtual inline int MinTopBlobs() const { return 1; }
      virtual inline int MaxTopBlobs() const { return 2; }

    protected:
      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top);

      virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
          const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

      /// The internal SoftmaxLayer used to map predictions to a distribution.
      shared_ptr<Layer<Dtype> > softmax_layer_;
      /// prob stores the output probability predictions from the SoftmaxLayer.
      Blob<Dtype> prob_;
      /// bottom vector holder used in call to the underlying SoftmaxLayer::Forward
      vector<Blob<Dtype>*> softmax_bottom_vec_;
      /// top vector holder used in call to the underlying SoftmaxLayer::Forward
      vector<Blob<Dtype>*> softmax_top_vec_;
      // sum of weights
      Dtype weights_sum_;
      Dtype dis_sigma;
      int count_point[15];
  };
}  // namespace caffe

#endif  // CAFFE_SOFTMAX_WITH_LOSS_LAYER_HPP_

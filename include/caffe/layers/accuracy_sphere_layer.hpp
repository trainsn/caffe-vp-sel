#ifndef CAFFE_ACCURACY_SPHERE_LAYER_HPP_
#define CAFFE_ACCURACY_SPHERE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the classification accuracy for a one-of-many
 *        classification task.
 */
/**
 * Author: Shi Neng
 * @brief Computes the average sphere classification accuracy. (category known by label)
 */
template <typename Dtype>
  class AccuracySphereLayer : public Layer<Dtype> {
    public:
      explicit AccuracySphereLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
      virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top);
      virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top);

      virtual inline const char* type(){
        return "AccuracySphere";
      }

      virtual inline int ExactNumBottomBlobs() const { return 2; }
      virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:

      virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top);


      /// @brief Not implemented -- AccuracySphereLayer cannot be used as a loss.
      virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
          const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        for (int i = 0; i < propagate_down.size(); ++i) {
          if (propagate_down[i]) { NOT_IMPLEMENTED; }
        }
      }

      float tol_angle_;
      int count_point[15];
  };
}  // namespace caffe

#endif  // CAFFE_ACCURACY_LAYER_HPP_

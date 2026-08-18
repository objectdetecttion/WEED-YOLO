# MIE-YOLO: A Multi-Scale Information Enhanced Weed Detection Algorithm for Precision Agriculture

## Abstract

As precision agriculture places higher demands on real-time field weed detection and recognition accuracy, this paper proposes a multi-scale information enhanced weed detection algorithm, MIE-YOLO (Multi-scale Information Enhanced), for precision agriculture. Based on the popular YOLO12 (You Only Look Once 12) model, this paper proposes four key improvements to address the challenges of small-scale, diverse morphologies, and strong background interference in field weeds. First, the MS-EIS (Multi-Scale-Edge Information Select) architecture is designed to effectively aggregate and select edge and texture information at different scales to enhance fine-grained feature representation. Next, the Add-CGLU (Additive-Convolutional Gated Linear Unit) pyramid network is proposed, which enhances the representational power and information transfer efficiency of multi-scale features through additive fusion and gating mechanisms. Finally, the DEC (Detail-Enhanced Convolution) detection head is introduced to enhance detail and refine the localization of small objects and fuzzy boundaries. To further improve the model's detection accuracy and generalization performance, the DS (Double Self-Knowledge Distillation) strategy is defined to perform double self-knowledge distillation within the entire network. Experimental results on the custom Weed dataset show that MIE-YOLO improves the F1 score by 1.9% and the mAP by 2.0%. Furthermore, it reduces computational parameters by 29.9%, FLOPs by 6.9%, and model size by 17.0%, achieving a runtime speed of 66.2 FPS. MIE-YOLO improves weed detection performance while maintaining a certain level of inference efficiency, providing an effective technical path and engineering implementation reference for intelligent field inspection and precise weed control in precision agriculture.

## Methodology

1. A new MS-EIS (Multi Scale-Edge Information Select) architecture was designed to effectively aggregate and select edge and texture information at different scales to enhance fine-grained feature expression.
2. A new Add-CGLU (Additive-Convolutional Gated Linear Unit) pyramid network was proposed to improve the representation capability and information transmission efficiency of multi-scale features through additive fusion and gating mechanisms.
3. A new DEC (Detail-Enhanced Convolution) detection head was developed to enhance details and refine positioning of small objects and fuzzy boundaries.
4. A new DS (Double Self-Knowledge Distillation) strategy was defined to perform double self-knowledge distillation on the entire network, further improving the model detection accuracy and generalization performance.

## Environment settings

Ensure the following dependencies are installed:

- torch >= 2.2.2+cu121
- torchvision >= 0.17.2+cu121
- CUDA >= 12.1.105
- cuDNN >= 8.9.7.29

Install other dependencies listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Datasets

- Weed (Custom Dataset)
- ImageWeeds (https://doi.org/10.1016/j.dib.2023.109691)
- VCD (https://doi.org/10.1016/j.dib.2022.108035)
- Weed-Crop (https://doi.org/10.1016/j.dib.2025.111486)

## License

This project is open-sourced under the MIT License.
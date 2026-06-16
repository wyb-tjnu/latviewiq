# No-reference Omnidirectional Image Quality Assessment Based on Viewport and Latitude-aware Distortion Adaptive Correction

## Paper Information
**Title**: No reference Omnidirectional Image Quality Assessment Based on Viewport and Latitude-aware Distortion Adaptive Correction

**Authors**: Bo Zhang*, Shucun Si, Cheng Zhang, Yibo Wang

**Affiliations**:
1* College of Computer and Information Engineering, Tianjin Normal University, Binshui West Road, Tianjin, 300087, China.
2 School of Electronics & Information Engineering, Tiangong Univercity, Binshui West Road, Tianjin, 300387, China.
3 Tianjin Key Laboratory of Optoelectronic Detection Technology & System, Tianjin, 300087, China.

**Corresponding Author**: Bo Zhang
**Contact E-mail**: tjnuzhangbo@163.com
**Author Emails**: sishucun2023@163.com; zhangcheng@tiangong.edu.cn; wangyibo2002@139.com

## News
- **2026-06-16**: We upload the model source code.


## Dataset Preparation
This paper evaluates the proposed method on two classic omnidirectional image quality assessment datasets: **OIQA** and **CVIQ**.

Due to the large storage size and copyright restrictions of the original datasets, we do not upload raw data in this repository.

**Dataset Download Links**:
- OIQA Dataset
- [CVIQ Dataset](https://github.com/sunwei925/CVIQDatabase)


## Train & Test

* Edit `Image_process/configuration.py` for implementation
* Run the following code to extract viewports

```bash
# python viewport_extracting.py 
```
* Edit `src/configs` and `configs` for implementation
* Run the file `src/trainer.py` and `src/test.py` for training and testing


## Citation

This paper is currently under peer review. The formal citation information will be updated after acceptance.
```bibtex
@article{zhang2026,
  title={No reference Omnidirectional Image Quality Assessment Based on Viewport and Latitude-aware Distortion Adaptive Correction},
  author={Zhang, Bo and Si, Shucun and Zhang, Cheng and Wang, Yibo},
  journal={Under Review},
  year={2026}
}
```

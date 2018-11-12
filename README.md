<h2 align="center">Implementing Global Illumination Algorithms on Nori</h2>


## Table of contents
- [Why Nori](#why-nori)
- [What's included](#whats-included)
- [Table of algorithms](#table-of-algorithms)
- [BDPT](#bdpt)
- [PM](#pm)
- [VCM](#vcm)
- [Thanks](#thanks)

## Why Nori
[Nori](https://github.com/wjakob/nori) is a simple ray tracer written in C++ for educational usage, and I learned rendering with it on course [Realistic Image Synthesis](https://www.ics.uci.edu/~shz/courses/cs295/)

## What's included

Within the download, you will find the following directories and files. Copy the files to corresponding directories in Nori, and it's good to use. The structure of files looks like this: 

```text
Rendering-On-Nori/
├── include/
|    └──nori/
|        └── nanoflann.hpp
└── src
     ├── bdpt.cpp
     ├── photon_mapping.cpp
     └── vcm.cpp
```
File `nanoflann.hpp` is an external file from [jlblancoc](https://github.com/jlblancoc/nanoflann/blob/master/include/nanoflann.hpp). Files `vcm.cpp` and `photon_mapping` use it to perform the knn search for photon estimation.

## Table of algorithms
- Bidirectional Path Tracing (BDPT) 
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`bdpt.cpp`
- Photon Mapping(PM)
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`photon_mapping.cpp`, `nanoflann.hpp`
- Vertex Connection and Merging (VCM) 
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`vcm.cpp`, `nanoflann.hpp`

## BDPT
BDPT is implemented with reference [Veach 1997 Robust Monte Carlo Methods for Light Transport Simulation](http://graphics.stanford.edu/papers/veach_thesis/).
## PM
Final Gathering as the photon mapping method, which combines photon estimation with direct illumination, is implemented instead of visualing the photon map, which has spots distributed throughout the scene.

[1]	Jensen, Henrik W., Realistic Image Synthesis Using Photon Mapping, A K Peters, Ltd., Massachusetts, 2001

[2] Zack Waters, Photon Mapping, https://web.cs.wpi.edu/~emmanuel/courses/cs563/write_ups/zackw/photon_mapping/PhotonMapping.html
## VCM
[VCM](http://www.smallvcm.com/) refers BDPT as Vertex Connection(VC), and PM as Vertex Merging(VM) in the paper. It uses multiple importance sampling(MIS) to integrate BDPT and PM. The most taxing and rewarding part is implementing the MIS, which is the essense of VCM. In this code, the basic MIS is realized, without reusing the recursive  sub-path weights. Thus, even for a simple scene, rendering is much slower than BDPT.

### Results
#### Diffuse surfaces
- Reference
<img src="https://www.ics.uci.edu/~zhanhanl/images/vcm/diffuse_ref.png" alt="Diffuse Refernce" width=320 height=240>
- VCM = VC + VM
<img src="https://www.ics.uci.edu/~zhanhanl/images/vcm/diffuse_vcm.png" alt="Diffuse VCM" width=320 height=240>&nbsp;=&nbsp;<img src="https://www.ics.uci.edu/~zhanhanl/images/vcm/diffuse_vc.png" alt="Diffuse VC" width=320 height=240>&nbsp;+&nbsp;<img src="https://www.ics.uci.edu/~zhanhanl/images/vcm/diffuse_vm.png" alt="Diffuse VM" width=320 height=240>


#### Caustic
- Reference

![cau ref](https://www.ics.uci.edu/~zhanhanl/images/vcm/caustic_ref.png =200x40)
- VCM

![cau vcm](https://www.ics.uci.edu/~zhanhanl/images/vcm/caustic_vcm.png =200x40)

## Thanks
Thanks to [Prof. Shuang Zhao](www.shuangz.com) for helping me understand the multiple importance sampling and BDPT concept.


 

# PULASki: Learning inter-rater variability using statistical distances to improve probabilistic segmentation
PULASki = **P**robabilistic **U**net with **L**oss **A**ssessed through **S**tatistical d**I**stances

The official code of the paper "PULASki: Learning inter-rater variability using statistical distances to improve probabilistic segmentation" (DOI: https://doi.org/10.1016/j.media.2025.103623, preprint: https://arxiv.org/abs/2312.15686)

The preliminary idea was presented at the SIAM Conference on Mathematics of Data Science in September 2022, within the talk titled "Model Geometry Uncertainty Quantification for Improved Hemodynamic Data Assimilation".

Then, the method and the initial results were presented at the ISMRM 2023 in June 2023, titled "Exploiting the inter-rater disagreement to improve probabilistic segmentation" (Program Number: 0810). Abstract available on [ResearchGate](https://www.researchgate.net/publication/370978766_Exploiting_the_inter-rater_disagreement_to_improve_probabilistic_segmentation) and the talk is available on [YouTube](https://www.youtube.com/watch?v=99ECgbWvME8) 

## Model Weights
The weights of the main Probabilistic UNet (PULASki + Baseline) models trained during this research, have been made publicly available on Huggingface, and they can be found in the collection: [https://huggingface.co/collections/soumickmj/pulaski-66d9d35dfef91c84d140de8d](https://huggingface.co/collections/soumickmj/pulaski-66d9d35dfef91c84d140de8d). The designations "VSeg" and "MSSeg" within the model names indicate that the respective model was trained for vessel segmentation and multiple sclerosis segmentation tasks, respectively. The PULASki models have "Hausdorff", "Sinkhorn" or "FID" in their names, indicating which statistical distance was used to train that particular model, while "Base" in their names signify that they are the baseline ProbUNet models (i.e. trained with the Focal-tversky loss or FTL). The model names also indicate whether they are 2D or 3D. 

The weights can be directly be used pulling from Huggingface with the updated version of this pipeline, or the weights can be downloaded using the AutoModel class from the transformers package, saved as a checkpoint, and then the path to this saved checkpoint can be supplied to the pipeline using "-load_path" argument.

Here's an example of how to use directly use weights from huggingface:
```bash
-load_huggingface soumickmj/PULASki_ProbUNet2D_Hausdorff_VSeg
```
Additional parameter "-load_huggingface" must be supplied along with the other desired paramters. Technically, this paramter can also be used to supply segmentation models other than the models used in DS6. 

Here is an example of how to save the weights locally (must be saved with .pth extension) and then use it with this pipeline:
```python
from transformers import AutoModel
modelHF = AutoModel.from_pretrained("soumickmj/PULASki_ProbUNet2D_Hausdorff_VSeg", trust_remote_code=True)
torch.save({'state_dict': modelHF.model.state_dict()}, "/path/to/checkpoint/model.pth")
```
To run this pipeline with these downloaded weights, the path to the checkpoint must then be passed as preweights_path, as an additional parameter along with the other desired parameters:
```bash
-load_path /path/to/checkpoint/model.pth
```

## Credits

If you like this repository, please click on Star!

If you use this approach in your research or use codes from this repository or the weights from Huggingface, please cite the following in your publications:

> [Chatterjee, S., Gaidzik, F., Sciarra, A., Mattern, H., Janiga, G., Speck, O., ... & Pathiraja, S. (2025). PULASki: Learning inter-rater variability using statistical distances to improve probabilistic segmentation. Medical Image Analysis, 103623.](https://authors.elsevier.com/a/1l5Ba4rfPmLeth)

BibTeX entry:

```bibtex
@article{chatterjee2025pulaski,
  title={PULASki: Learning inter-rater variability using statistical distances to improve probabilistic segmentation},
  author={Chatterjee, Soumick and Gaidzik, Franziska and Sciarra, Alessandro and Mattern, Hendrik and Janiga, G{\'a}bor and Speck, Oliver and N{\"u}rnberger, Andreas and Pathiraja, Sahani},
  journal={Medical Image Analysis},
  pages={103623},
  year={2025},
  publisher={Elsevier}
}

```

The original codebase is from the DS6 project (https://github.com/soumickmj/DS6). Initial code was also published as a part of that same repo (Branch: [ProbVSeg](https://github.com/soumickmj/DS6/tree/ProbVSeg)).

<!-- please cite the following in your publications: -->

<!-- > [Soumick Chatterjee, Kartik Prabhu, Mahantesh Pattadkal, Gerda Bortsova, Chompunuch Sarasaen, Florian Dubost, Hendrik Mattern, Marleen de Bruijne, Oliver Speck, Andreas Nürnberger: DS6, Deformation-aware Semi-supervised Learning: Application to Small Vessel Segmentation with Noisy Training Data (arXiv:2006.10802, June 2020)](https://arxiv.org/abs/2006.10802) -->

BibTeX entry (for the original DS6 project):

```bibtex
@article{chatterjee2022ds6,
          AUTHOR = {Chatterjee, Soumick and Prabhu, Kartik and Pattadkal, Mahantesh and Bortsova, Gerda and Sarasaen, Chompunuch and Dubost, Florian and Mattern, Hendrik and de Bruijne, Marleen and Speck, Oliver and Nürnberger, Andreas},
          TITLE = {DS6, Deformation-Aware Semi-Supervised Learning: Application to Small Vessel Segmentation with Noisy Training Data},
          JOURNAL = {Journal of Imaging},
          VOLUME = {8},
          YEAR = {2022},
          NUMBER = {10},
          ARTICLE-NUMBER = {259},
          URL = {https://www.mdpi.com/2313-433X/8/10/259},
          ISSN = {2313-433X},
          DOI = {10.3390/jimaging8100259}
}

```
Thank you so much for your support.


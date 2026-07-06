# Citations

## Concussion Speech-Abnormality Model

The bundled `backend/models/concussion_speech` artifacts are the runtime output
of the internal `pilot_full` speech-abnormality model. The model is a
research-only speech classifier, not a concussion, dysarthria, or dysphonia
diagnostic system.

### Training Datasets

- **TORGO Database**: acoustic and articulatory speech from speakers with
  dysarthria and matched controls. Used for `dysarthria_like` and `normal`
  training examples.
  - Site: [The TORGO Database: Acoustic and articulatory speech from speakers with dysarthria](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html)
  - Suggested citation: Rudzicz, F., Namasivayam, A. K., & Wolff, T. (2012). The TORGO database of acoustic and articulatory speech from speakers with dysarthria. *Language Resources and Evaluation*, 46, 523-541.

- **VOICED Database v1.0.0**: healthy and pathological sustained-vowel voice
  samples from PhysioNet. Used for `dysphonia_like` and `normal` training
  examples.
  - Site: [VOICED Database v1.0.0](https://physionet.org/content/voiced/1.0.0/)
  - Suggested citation: Cesari, U., De Pietro, G., Marciano, E., Niri, C., Sannino, G., & Verde, L. (2018). A new database of healthy and pathological voices. *Computers & Electrical Engineering*, 68, 310-321.
  - PhysioNet citation: Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*, 101(23), e215-e220.

### Feature Backbone And Classifier

- **WavLM Base**: frozen speech embedding backbone configured as
  `microsoft/wavlm-base` in `backend/models/concussion_speech/config.json`.
  Runtime audio is resampled to 16 kHz before embedding.
  - Model card: [microsoft/wavlm-base on Hugging Face](https://huggingface.co/microsoft/wavlm-base)
  - Paper: [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900)
  - Suggested citation: Chen, S., Wang, C., Chen, Z., Wu, Y., Liu, S., Chen, Z., et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing. *IEEE Journal of Selected Topics in Signal Processing*.

- **Classifier**: scikit-learn logistic regression with class balancing and
  calibration, saved in `backend/models/concussion_speech/model.joblib`.
  - API reference: [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### Runtime Labels

The checked-in model returns these research labels only:

- `normal`
- `dysarthria_like`
- `dysphonia_like`
- `low_audio_quality`

The labels are dataset-derived research categories. They are not medical
diagnoses and must not be presented as concussion detection.

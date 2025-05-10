## 

## Introduction: 

This study addresses the computational inefficiencies and numerical instabilities inherent in traditional BLS when handling complex tasks in dynamic environments. To mitigate these challenges, we propose an enhanced version of BLS incorporating QR factorization (QRF), referred to as QRBLS, which is known for improving numerical stability. This framework replaces the traditional method of computing output weights, which typically relies on the Moore-Penrose pseudoinverse. The primary contribution of this research is the integration of QRF into the BLS architecture, thereby improving stability when processing large-scale datasets. QRBLS also features a dynamic updating mechanism that adjusts model parameters efficiently with new data, enabling continuous learning without the need for full model reevaluation. Additionally, a time-dependent structure (TDS) enhances the model’s responsiveness to temporal data changes, increasing its utility in dynamic environments. Validation through numerical experiments demonstrated that QRBLS outperformed traditional BLS, exhibiting superior stability and adaptability in handling data anomalies and rapid updates. The integration of QRF and TDS significantly improves the adaptability and computational efficiency of BLS, providing a robust solution for large-scale, dynamic AI applications. QRBLS effectively addresses challenges related to numerical instability and continuous learning, offering practical improvements in real-world settings.

## Environment Setup:

Before using this library, please ensure that you have the following essential packages and their corresponding versions installed.

<center>

| Package            | Version       |
|--------------------|---------------|
| numpy              | 1.21.6        |
| matplotlib         | 3.2.2         |
| scikit-learn       | 0.22.1        |
| scikit-multiflow   | 0.5.3         |
| pandas             | 1.2.3         |
| scipy              | 1.7.3         |

</center>

## Note

Your feedback, questions, and contributions are invaluable to us. Whether you have suggestions for improvements, encounter issues, or wish to collaborate on enhancements, we welcome your participation. Together, we can continue to refine and expand this toolkit to empower researchers, practitioners, and enthusiasts in the field.

Please feel free to reach out to us via email with [Chen Li](mailto:yan12@tsinghua.org.cn) and [Zeyi Liu](mailto:liuzy21@mails.tsinghua.edu.cn). Here's to a fruitful learning journey!


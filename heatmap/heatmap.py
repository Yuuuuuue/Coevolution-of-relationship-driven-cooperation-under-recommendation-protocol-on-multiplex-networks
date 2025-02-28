import matplotlib.pyplot as plt
import numpy as np

f_3 = [
    [0.9551554, 0.8915992, 0.6881657999999998, 0.44055999999999995, 0.22294319999999998, 0.1340818, 0.0699778, 0.046095199999999996, 0.0204516, 0.013232, 0.0100628, 0.0047962, 0.0028066000000000002, 0.0008352000000000001, 0.0003562, 0.0001322, 0.00022580000000000004, 8.2e-06, 6.199999999999999e-06, 0.0, 0.0],
    [0.9436466, 0.8746374, 0.6445122000000001, 0.3467314, 0.19640899999999997, 0.08689920000000001, 0.0411268, 0.0231956, 0.015546599999999999, 0.007980000000000001, 0.0035605999999999997, 0.0025595999999999995, 0.0021332, 0.00030940000000000004, 0.0, 9.28e-05, 0.00014179999999999998, 0.0, 0.0, 0.0, 0.0],
    [0.944011, 0.871223, 0.6445104, 0.38439, 0.1916734, 0.10188839999999999, 0.04873159999999999, 0.029601, 0.0161116, 0.0088478, 0.0059466, 0.0034061999999999994, 0.0010758, 0.0010042, 0.0002566, 0.00013319999999999996, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.9505132000000001, 0.883697, 0.699273, 0.38836, 0.19607799999999997, 0.1088664, 0.04923640000000001, 0.0330022, 0.017335, 0.0079574, 0.0063604, 0.0037281999999999997, 0.0010324, 0.0007572000000000001, 0.00011379999999999997, 0.00023760000000000003, 0.00012979999999999996, 0.0, 0.0, 0.0, 0.0],
    [0.945779, 0.8806208000000001, 0.6998202, 0.4103264, 0.2255562, 0.131304, 0.057359799999999996, 0.034489799999999994, 0.017608400000000003, 0.011177399999999999, 0.0089718, 0.0044101999999999995, 0.0016225999999999999, 0.0002032, 0.0002694, 0.0004432, 0.0, 0.00011699999999999998, 0.0, 0.0, 0.0],
    [0.9524278000000002, 0.8809116, 0.6996353999999999, 0.45130700000000007, 0.2271446, 0.1290548, 0.06831179999999999, 0.0399714, 0.0227168, 0.012981600000000003, 0.0056038, 0.0034416000000000004, 0.0015116, 0.0008624000000000001, 0.0005244, 0.00033279999999999996, 0.0001152, 0.00011759999999999999, 0.00013319999999999999, 0.0, 0.0],
    [0.9474906, 0.8866232000000001, 0.6926988000000001, 0.4586656, 0.2544914, 0.14451799999999998, 0.0811144, 0.045401000000000004, 0.028150199999999997, 0.0151328, 0.0088914, 0.0055988, 0.0035787999999999996, 0.0016402, 0.0008818000000000001, 0.0006226, 0.0003286, 0.00012200000000000002, 0.0, 0.0, 0.0],
    [0.9529358000000001, 0.8848622, 0.7439361999999999, 0.44745579999999996, 0.2900964, 0.1659908, 0.0912196, 0.056166400000000005, 0.0327684, 0.019627400000000003, 0.010322, 0.0067578000000000004, 0.0035175999999999996, 0.002362, 0.0007204000000000001, 0.00042440000000000007, 0.0006934000000000002, 0.00018460000000000004, 0.0, 0.00020720000000000002, 0.0],
    [0.953022, 0.8798767999999998, 0.7132404, 0.5039198, 0.31136700000000006, 0.18757400000000005, 0.09992459999999999, 0.0514676, 0.0375422, 0.022623800000000003, 0.014980600000000002, 0.0070742, 0.0052418, 0.0031836000000000004, 0.0015440000000000005, 0.00097, 0.00047560000000000007, 0.0, 0.0, 1.0600000000000002e-05, 0.0],
    [0.9496772, 0.8862252, 0.7412258, 0.5196506, 0.3205854, 0.18862839999999997, 0.10691060000000001, 0.06701860000000001, 0.034877000000000005, 0.026440400000000003, 0.015361599999999998, 0.008605799999999999, 0.0047688, 0.0032648, 0.0020412, 0.0007354, 0.0005938, 0.0001286, 0.000209, 0.0, 7.74e-05],
    [0.9493272, 0.9026352, 0.7655356000000001, 0.566882, 0.33026679999999997, 0.20123759999999996, 0.13268139999999998, 0.077484, 0.0450004, 0.028295999999999998, 0.0162472, 0.0114226, 0.0069234, 0.0036098, 0.0034409999999999996, 0.0010556, 0.000908, 0.0006507999999999999, 0.00034999999999999994, 0.00046499999999999997, 0.0],
    [0.9516368, 0.9017390000000001, 0.7568768, 0.5731462, 0.3491466, 0.20819600000000002, 0.123311, 0.08012520000000001, 0.0487476, 0.030903200000000002, 0.01973, 0.010247, 0.007696799999999999, 0.003614, 0.0037992000000000004, 0.0018792000000000001, 0.0012051999999999998, 0.00039440000000000005, 0.00043380000000000003, 6.500000000000001e-05, 0.0],
    [0.9546247999999999, 0.9087982000000001, 0.8023596000000001, 0.6067846, 0.37749279999999996, 0.2442494, 0.1451602, 0.09279519999999998, 0.0559182, 0.037408000000000004, 0.0231482, 0.015558, 0.012043599999999998, 0.005812, 0.003846, 0.0025322, 0.0013742, 0.00047979999999999995, 0.0006439999999999998, 7.520000000000001e-05, 8.999999999999998e-05],
    [0.945087, 0.9068352000000001, 0.8059307999999998, 0.6335000000000001, 0.40424, 0.29019280000000003, 0.16659960000000001, 0.1028534, 0.07011099999999999, 0.0518916, 0.025492400000000002, 0.016189799999999997, 0.010505200000000001, 0.0068754, 0.0047954, 0.0035028000000000004, 0.0017014, 0.0011021999999999998, 0.0010547999999999998, 0.0008062, 0.0],
    [0.9461804, 0.9046417999999999, 0.8090006, 0.6464279999999999, 0.4346178, 0.28632080000000004, 0.18586640000000001, 0.10725319999999999, 0.0789564, 0.051737599999999995, 0.034177400000000004, 0.0219958, 0.015033399999999999, 0.0097364, 0.0061660000000000005, 0.004895, 0.002831, 0.0017487999999999998, 0.0005576, 0.0006653999999999999, 0.0004054],
    [0.9543904000000001, 0.9111848, 0.8066086000000001, 0.6752188000000001, 0.4579058, 0.3322646, 0.1992166, 0.132645, 0.0795246, 0.054486999999999994, 0.040848600000000006, 0.027825199999999998, 0.015867, 0.011758, 0.008146800000000001, 0.0051452, 0.0037978, 0.0027264, 0.0007326, 0.0013122000000000001, 0.0008604000000000001],
    [0.9502202, 0.9048117999999999, 0.8396429999999999, 0.7004784000000001, 0.502202, 0.3586768, 0.2111402, 0.1391608, 0.09481880000000001, 0.07094519999999999, 0.0423374, 0.0316742, 0.0270084, 0.014441199999999998, 0.0080914, 0.0077442, 0.0040068, 0.0026613999999999995, 0.0015745999999999998, 0.0011953999999999999, 0.0011966],
    [0.9516326, 0.9248833999999999, 0.8299652000000001, 0.7066241999999999, 0.536609, 0.4009664, 0.25588720000000004, 0.166934, 0.123712, 0.0792474, 0.056825799999999996, 0.0353452, 0.022423000000000002, 0.0163036, 0.013756200000000001, 0.0074428, 0.0053144, 0.0032021999999999997, 0.0028798, 0.00255, 0.0013426],
    [0.9541764, 0.9205208, 0.83976, 0.7202485999999999, 0.5858744, 0.4118935999999999, 0.2776892, 0.1771568, 0.136923, 0.0892666, 0.057470999999999994, 0.0476718, 0.027721799999999998, 0.023516399999999996, 0.014746199999999998, 0.0106296, 0.006773199999999999, 0.0057621999999999994, 0.003652, 0.0030681999999999997, 0.0017812],
    [0.9514539999999999, 0.9230687999999999, 0.8482993999999999, 0.7318036000000001, 0.5854494, 0.468353, 0.336528, 0.2153366, 0.15854659999999998, 0.1073756, 0.0782742, 0.05152060000000001, 0.037092, 0.0217402, 0.015932599999999998, 0.011882000000000002, 0.00904, 0.0077728, 0.0039968, 0.0046616, 0.0022064],
    [0.9487979999999999, 0.9233526000000001, 0.8701792, 0.7675607999999999, 0.6505372, 0.502835, 0.3218664, 0.242917, 0.1549288, 0.125019, 0.08568719999999999, 0.05399940000000001, 0.0389742, 0.030034, 0.022588800000000006, 0.0181492, 0.0132442, 0.005349999999999999, 0.006055, 0.0052088, 0.003125]
]

f_4 = [
    [0.9876136, 0.9680984000000001, 0.8804695999999999, 0.588478, 0.27216419999999997, 0.1300138, 0.0738246, 0.039573, 0.0254278, 0.014905199999999999, 0.0071668, 0.0026983999999999997, 0.002072, 0.0009642, 0.0004848, 0.00047420000000000003, 8.979999999999999e-05, 0.00028839999999999996, 0.0, 0.0, 0.0],
    [0.9860052000000001, 0.9685061999999999, 0.9023058, 0.6563631999999999, 0.32174099999999994, 0.15518259999999998, 0.077812, 0.0389968, 0.0275402, 0.0156274, 0.0094118, 0.0049388, 0.0030031999999999997, 0.0013088, 0.0009558, 0.0007794000000000001, 0.0001976, 0.0001926, 0.0, 0.0, 0.0],
    [0.9848176, 0.9613471999999998, 0.9092848000000001, 0.6802537999999998, 0.3666778, 0.1559206, 0.07962659999999999, 0.048630599999999996, 0.0278738, 0.017239400000000002, 0.009706600000000001, 0.0059266, 0.0031504000000000002, 0.0023548, 0.0009206, 0.0006006, 0.0002418, 0.0, 0.0, 0.00010059999999999999, 0.0],
    [0.9838667999999998, 0.9713824, 0.9006008, 0.7042852, 0.35641300000000004, 0.16494299999999998, 0.09466080000000002, 0.056308000000000004, 0.0353148, 0.019204600000000002, 0.012170800000000002, 0.006744999999999998, 0.004924, 0.0030655999999999995, 0.0013817999999999999, 0.0005326, 0.0001906, 0.000234, 0.0001946, 0.0, 0.0],
    [0.9865366, 0.9692537999999999, 0.9009316000000002, 0.7664023999999999, 0.38556179999999995, 0.1918028, 0.09726199999999999, 0.0636076, 0.03481360000000001, 0.022417600000000003, 0.0126452, 0.007722199999999999, 0.0052868, 0.0046045999999999995, 0.0015775999999999997, 0.0015918, 0.0006196, 7.859999999999999e-05, 0.0004812, 7.96e-05, 0.0],
    [0.9853216, 0.9672918, 0.9138512, 0.740071, 0.4574356, 0.2265898, 0.10568200000000001, 0.066609, 0.0417084, 0.0285616, 0.0122072, 0.008522, 0.0049602, 0.0030224, 0.0022704, 0.0010424, 0.0008032, 0.0004000000000000001, 0.00018080000000000003, 0.00013039999999999997, 0.0],
    [0.9817188, 0.9639804000000001, 0.932667, 0.7563806, 0.4609218, 0.2262942, 0.12023620000000002, 0.0655382, 0.0477152, 0.0283094, 0.0175592, 0.009316999999999999, 0.006352, 0.0044724, 0.0033697999999999996, 0.001879, 0.0010161999999999999, 0.0006201999999999999, 0.0003694, 0.00011839999999999998, 0.00010779999999999999],
    [0.9840523999999999, 0.9701814, 0.9217238000000001, 0.7872013999999998, 0.5239119999999999, 0.2397232, 0.13906780000000002, 0.087471, 0.0529798, 0.030068, 0.020245799999999998, 0.0143502, 0.0070723999999999995, 0.0051470000000000005, 0.0031911999999999995, 0.0024164, 0.0020152, 0.0009107999999999999, 0.0012193999999999998, 0.000746, 0.00010419999999999998],
    [0.9883102000000001, 0.9689926, 0.9303855999999999, 0.8244614000000001, 0.5377504000000001, 0.284683, 0.1404054, 0.10700399999999999, 0.0577466, 0.0341998, 0.020723000000000002, 0.0159456, 0.0113554, 0.004908, 0.004845599999999999, 0.0021417999999999997, 0.0019106000000000001, 0.0004964, 0.00040080000000000004, 0.00022820000000000004, 9.68e-05],
    [0.9849705999999999, 0.9729642000000002, 0.9438453999999998, 0.8355817999999999, 0.5768800000000001, 0.3511202, 0.154781, 0.10470800000000001, 0.0572396, 0.039580000000000004, 0.0271854, 0.0194238, 0.014181599999999999, 0.0070016, 0.004966, 0.0034395999999999997, 0.0018978, 0.0010729999999999997, 0.0007662000000000001, 0.0003656, 0.0001074],
    [0.9856224, 0.9730156, 0.9384136, 0.8541866, 0.6107182, 0.3381668, 0.18954000000000001, 0.10705339999999999, 0.07370379999999999, 0.0516606, 0.028486, 0.0223192, 0.0126958, 0.008201199999999999, 0.0048014, 0.0039944, 0.0021533999999999998, 0.0013790000000000002, 0.0005475999999999999, 0.00021780000000000004, 0.0005602],
    [0.9825694, 0.9792988, 0.9427486, 0.8496994, 0.6361636, 0.3620194, 0.212686, 0.12101560000000001, 0.07900060000000002, 0.0532362, 0.034787, 0.0255056, 0.0179746, 0.014760200000000001, 0.0079106, 0.005690999999999999, 0.0030642000000000004, 0.0030055999999999998, 0.0013464000000000002, 0.0006296, 0.000442],
    [0.9873186000000002, 0.9749232000000001, 0.9493504, 0.8704854000000001, 0.7022515999999999, 0.42531099999999994, 0.21510579999999999, 0.15399059999999998, 0.08228619999999999, 0.0613184, 0.0465146, 0.0258972, 0.018536200000000003, 0.01299, 0.0066282, 0.0069478, 0.0033038, 0.00301, 0.0016656, 0.001089, 0.0008292000000000001],
    [0.9845016000000001, 0.9775856, 0.9487078, 0.8868946, 0.6955854000000001, 0.452064, 0.2695716, 0.1739504, 0.09412980000000001, 0.0615324, 0.050046400000000005, 0.0324814, 0.020642399999999998, 0.013757000000000002, 0.010032000000000001, 0.0065639999999999995, 0.0046451999999999995, 0.003928200000000001, 0.0014880000000000002, 0.0009848, 0.0010253999999999999],
    [0.9851034, 0.9796935999999999, 0.9533326000000001, 0.8773332, 0.7507754, 0.5522121999999999, 0.2936474, 0.17890099999999998, 0.1269378, 0.0799474, 0.05138679999999999, 0.0339472, 0.027182400000000002, 0.017237199999999998, 0.0123854, 0.0091488, 0.0084299, 0.00610000012, 0.00349211, 0.001118293,0.0010000021],
    [0.987225, 0.9773558, 0.9500336, 0.9098864, 0.8054812, 0.5557588, 0.36506420000000006, 0.22257159999999998, 0.13886140000000002,  0.08855360000000001, 0.06083340000000001, 0.046164000000000004, 0.027921, 0.021070199999999997, 0.0172498, 0.010826,0.0093299, 0.00740646, 0.00400023, 0.002118293, 0.0199923],
    [0.9833905999999999, 0.9753054, 0.9550941999999999, 0.9189134, 0.8307312000000002, 0.6042184, 0.38479299999999994, 0.269034, 0.147944, 0.10353259999999999, 0.0639384, 0.048839600000000004, 0.0360874, 0.0248848, 0.015204200000000001, 0.014390999999999998, 0.0081008, 0.0071272, 0.006187400000000001, 0.0042722, 0.002391],
    [0.985189, 0.9764136000000001, 0.9582896, 0.9286416, 0.8287336, 0.6436918, 0.43464479999999994, 0.2694506, 0.1830884, 0.12484459999999999, 0.07946340000000002, 0.05798999999999999, 0.043571, 0.0348108, 0.023782, 0.020736400000000002, 0.0127788, 0.0088936, 0.0062086, 0.0048046, 0.0036514000000000004],
    [0.9851214, 0.9782894, 0.9627578, 0.934777, 0.869906, 0.7153776, 0.4921264, 0.3078438, 0.1877714, 0.12677639999999998, 0.0862992, 0.0753942, 0.047636200000000004, 0.037911600000000004, 0.0300062, 0.0204774, 0.015244, 0.010718, 0.0068882, 0.006078600000000001, 0.0048],
    [0.984975, 0.9784430000000001, 0.9616499999999999, 0.9391267999999999, 0.870093, 0.7708911999999999, 0.5751684, 0.3919858, 0.2366096, 0.1474608, 0.1119284, 0.0728754, 0.055040200000000004, 0.043532799999999996, 0.033665, 0.025345, 0.0194258, 0.015086599999999999, 0.010502000000000001, 0.0063338000000000005, 0.0043457999999999995],
    [0.9838508, 0.981399, 0.9670856000000001, 0.9439596, 0.8848158, 0.788224, 0.6336879999999999, 0.43938499999999997, 0.24797939999999996, 0.18842840000000002, 0.1345792, 0.10963200000000001, 0.0774022, 0.0447396, 0.039675800000000004, 0.036776, 0.0235356, 0.01872, 0.0141134, 0.0102636, 0.007592000000000001]
]

f_6 = [[0.9966932000000002, 0.9923384, 0.9806922, 0.9350504000000001, 0.7802180000000001, 0.5464648, 0.1843582, 0.11022680000000001, 0.0615008, 0.039047599999999995, 0.026764, 0.015503, 0.010046400000000002, 0.0049932, 0.005304, 0.0038618, 0.0017118, 0.0007582, 0.0008482, 0.00045339999999999996, 0.0002658],
       [0.9959138, 0.9931478, 0.9845112, 0.9355658, 0.8015237999999999, 0.5468276, 0.2131512, 0.1123048, 0.0642434, 0.042242999999999996, 0.02803, 0.018233, 0.0114934, 0.0080872, 0.0055975999999999995, 0.0020528, 0.0011593999999999999, 0.000547, 0.0005192, 0.00031120000000000003, 0.000215],
       [0.9947030000000001, 0.9921540000000002, 0.9842282, 0.9442400000000001, 0.8343827999999999, 0.6649956, 0.242029, 0.12156739999999999, 0.074431, 0.04933680000000001, 0.030404999999999998, 0.0199504, 0.012724, 0.0078488, 0.006418200000000001, 0.0041352, 0.002277, 0.001781, 0.0007058, 0.0006748, 0.0003774],
       [0.9965122000000001, 0.9931458000000001, 0.9861338000000001, 0.9531332, 0.844107, 0.692697, 0.30672520000000003, 0.11973220000000001, 0.078963, 0.0600746, 0.035910599999999994, 0.0210662, 0.015542, 0.0096326, 0.0057858, 0.0040436000000000005, 0.0028637999999999997, 0.0018284, 0.0006206, 0.0003892, 0.0006332],
       [0.9966472000000001, 0.9947084, 0.9872552000000002, 0.9472068000000001, 0.8596152, 0.7189418000000001, 0.3531796, 0.15821819999999998, 0.0827506, 0.0578524, 0.0432522, 0.025964400000000002, 0.0157724, 0.012749000000000002, 0.007842199999999999, 0.0046136, 0.003295, 0.0021958, 0.0014568, 0.0009012, 0.0005892],
       [0.9968904000000001, 0.9955118000000002, 0.988471, 0.960391, 0.8644312, 0.7213424, 0.4475502, 0.17428199999999996, 0.09795440000000001, 0.061593, 0.04316179999999999, 0.031012599999999998, 0.020295400000000005, 0.0122478, 0.0084878, 0.0060192, 0.0027308, 0.0029504, 0.00179, 0.0014416000000000001, 0.0008760000000000002],
       [0.9964044000000001, 0.9919602000000001, 0.9896914000000001, 0.9674448, 0.8797066000000001, 0.7607831999999999, 0.5088391999999999, 0.1790298, 0.11087259999999999, 0.074458, 0.0433776, 0.03136, 0.022232599999999998, 0.0166748, 0.0096958, 0.007902800000000001, 0.0057436, 0.0018460000000000002, 0.0021658, 0.0021566, 0.0012332],
       [0.997868, 0.9941939999999999, 0.9901471999999998, 0.9558546000000001, 0.8857622, 0.7849613999999999, 0.5321606, 0.22971819999999998, 0.1194764, 0.0772298, 0.0498532, 0.034341800000000006, 0.0246364, 0.015383199999999996, 0.0093464, 0.0077507999999999995, 0.005807400000000001, 0.0048768, 0.0036048, 0.0014438, 0.0019168000000000002],
       [0.997164, 0.994536, 0.9875396, 0.9740238000000001, 0.9151378, 0.7898954, 0.6373118000000001, 0.2967898, 0.16817720000000003, 0.09692260000000001, 0.0638642, 0.039723, 0.030356, 0.0200486, 0.0129724, 0.0094544, 0.0093842, 0.0043276, 0.002317, 0.0019782, 0.0014912],
       [0.9959616, 0.9944372, 0.9854022, 0.9776511999999999, 0.9227294, 0.8050809999999999, 0.6968733999999999, 0.3077472, 0.1776252, 0.1094968, 0.0694168, 0.04907740000000001, 0.0335238, 0.027739600000000003, 0.019378199999999998, 0.0114202, 0.0105698, 0.0031287999999999997, 0.0059216, 0.0027256000000000003, 0.0010077999999999999],
       [0.9962287999999999, 0.9935102, 0.9906540000000001, 0.9730509999999998, 0.9109248, 0.8220145999999999, 0.7037767999999999, 0.4407012, 0.1805284, 0.1081614, 0.0793774, 0.05807280000000001, 0.0383316, 0.026311400000000002, 0.0201548, 0.014664199999999999, 0.0084256, 0.0066782000000000005, 0.0048002, 0.0043574, 0.002817],
       [0.99715, 0.9947942000000001, 0.9900699999999999, 0.9825212000000001, 0.932541, 0.8508356, 0.7337261999999999, 0.5261832, 0.23199220000000004, 0.14445139999999998, 0.08335059999999998, 0.0626072, 0.043792, 0.0325346, 0.023696799999999997, 0.015889, 0.0116388, 0.011175199999999998, 0.005333000000000001, 0.004458999999999999, 0.0042174],
       [0.9945068, 0.9963713999999999, 0.9913822, 0.9776718, 0.9420053999999999, 0.8602334000000001, 0.7626069999999998, 0.5541006, 0.310015, 0.168678, 0.1043082, 0.0721192, 0.0497002, 0.0356842, 0.025085000000000007, 0.019440600000000002, 0.016679000000000003, 0.009604399999999999, 0.006794, 0.0050127999999999995, 0.0030279999999999994],
       [0.9964764000000002, 0.9958250000000001, 0.9937147999999999, 0.9807204, 0.9559788, 0.8670434, 0.7729798, 0.6732266000000001, 0.37098240000000005, 0.19264420000000002, 0.127491, 0.07708699999999999, 0.0625898, 0.044572200000000006, 0.029679, 0.025705999999999996, 0.016204399999999997, 0.0123508, 0.008973, 0.006778200000000001, 0.0050295999999999995],
       [0.9960692, 0.9945376, 0.9899128, 0.9857222, 0.9580592000000001, 0.8931208, 0.7939454, 0.6921634, 0.43396860000000004, 0.2489222, 0.16673839999999998, 0.088585, 0.06823459999999999, 0.05017939999999999, 0.0363038, 0.0256446, 0.0241248, 0.0139006, 0.012212, 0.0083416, 0.0056384],
       [0.9964542000000001, 0.9942046000000001, 0.9906691999999999, 0.9818566, 0.9770768, 0.9103749999999999, 0.8190791999999999, 0.7435684, 0.6211906, 0.25371, 0.16776139999999998, 0.10719200000000001, 0.068377, 0.0588926, 0.0373906, 0.03730000000000001, 0.026997200000000002, 0.015784199999999998, 0.015288000000000001, 0.0102888, 0.0083938],
       [0.9962774, 0.9943634, 0.9919170000000002, 0.9840974, 0.9645590000000002, 0.934758, 0.854876, 0.7834866000000001, 0.6401194000000001, 0.3112408, 0.19249940000000001, 0.1256332, 0.09136980000000001, 0.061807, 0.0576306, 0.0390978, 0.029573400000000003, 0.0199634, 0.0180916, 0.013694999999999999, 0.011019600000000001],
       [0.9966646, 0.9950026000000001, 0.9922496000000001, 0.9892730000000001, 0.9682578000000001, 0.9347642, 0.8671341999999999, 0.7820623999999999, 0.6971644, 0.45175979999999993, 0.2638204, 0.1624646, 0.11282699999999998, 0.0811172, 0.057475399999999996, 0.041094399999999996, 0.030506399999999996, 0.029476, 0.022756, 0.0174056, 0.010827799999999999],
       [0.9960005999999999, 0.9955272000000001, 0.9914673999999999, 0.9885837999999998, 0.9773068, 0.9401074, 0.8949636000000001, 0.823522, 0.7409280000000001, 0.5842288, 0.35951099999999997, 0.1848798, 0.1295544, 0.09850120000000001, 0.06630680000000001, 0.050396, 0.04354799999999999, 0.031062799999999998, 0.0286824, 0.0208846, 0.015195199999999999],
       [0.9964846, 0.9942310000000001, 0.9940450000000002, 0.9889009999999999, 0.9785993999999999, 0.9583012,0.8961887999999999, 0.8327844, 0.7419988000000001, 0.6756299999999998, 0.34260779999999996, 0.2330048,0.1627588, 0.11610240000000001, 0.0755224, 0.0663958, 0.050657799999999996, 0.0385166, 0.025809199999999997, 0.0256154, 0.0225184],
       [0.9968386000000001, 0.9953942000000001, 0.9941402, 0.9895668000000001, 0.9841188, 0.9632512, 0.9265706, 0.843637, 0.7991166, 0.7009237999999999, 0.5059416, 0.2700888, 0.2042088, 0.1253356, 0.1079952, 0.0765292, 0.0579602, 0.048014799999999996, 0.03802, 0.030724, 0.027042999999999998]
]

f_WS = [[0.9972213999999999, 0.9955018000000001, 0.9920614000000001, 0.9904604, 0.9787997999999999, 0.9323821999999998, 0.8700595999999998, 0.7921156, 0.7145006, 0.6577647999999999, 0.5738234, 0.41871880000000006, 0.2321542, 0.1498544, 0.1326426, 0.0730806, 0.0635584, 0.0589564, 0.0406358, 0.032675, 0.0253092],
        [0.9965153999999999, 0.995703, 0.9940612, 0.9776774, 0.9750176, 0.9523426, 0.8865801999999999, 0.8036536, 0.7414046000000001, 0.6769748000000001, 0.5758696, 0.4585208, 0.2482172, 0.17625619999999997, 0.126927, 0.0962216, 0.0792584, 0.06065160000000001, 0.0445174, 0.036949800000000005, 0.028654200000000005],
        [0.9965733999999999, 0.9944491999999999, 0.9929832000000001, 0.9886097999999999, 0.981353, 0.956109, 0.8971108, 0.8036032, 0.7583698000000001, 0.6910656000000001, 0.6107872000000001, 0.5655534, 0.32594880000000004, 0.19659100000000002, 0.1373228, 0.091563, 0.0752748, 0.060586400000000006, 0.0438872, 0.0383442, 0.0263028],
        [0.9964084, 0.9957264000000001, 0.993824, 0.9879612, 0.9785744, 0.9608188, 0.9006698, 0.82406, 0.7515172, 0.7122932, 0.6363346, 0.5791952, 0.45239240000000003, 0.21408760000000002, 0.1553352, 0.1046406, 0.09320260000000001, 0.06719979999999999, 0.05595979999999999, 0.050694800000000005, 0.0400088],
        [0.9961474, 0.9933735999999999, 0.9914429999999999, 0.9916693999999999, 0.9859962, 0.9678752, 0.9217348, 0.8583314, 0.7707616, 0.7149962, 0.663818, 0.6281606, 0.47217739999999997, 0.2143562, 0.221718, 0.1381264, 0.1005688, 0.078417, 0.06411019999999999, 0.051059, 0.052396200000000004],
        [0.9973848, 0.9959212, 0.9961666, 0.9835864, 0.9849277999999998, 0.9707517999999999, 0.933554, 0.8580102,0.7595224, 0.7217491999999999, 0.6713803999999999, 0.6086998, 0.5318406, 0.2922778, 0.2352684, 0.1467466,0.11453680000000001, 0.0863298, 0.061500599999999996, 0.050842, 0.05183120000000001],
        [0.9961983999999999, 0.9948950000000001, 0.993776, 0.992869, 0.9928702000000001, 0.9771878, 0.928068, 0.8686440000000001, 0.8226458, 0.7448005999999999, 0.6826926000000002, 0.650152, 0.5774482000000001, 0.3586486, 0.2968592, 0.2075812, 0.149078, 0.0948482, 0.0714462, 0.0635228, 0.0537162],
        [0.9966106, 0.9950945999999999, 0.9942601999999999, 0.9948456, 0.9920942, 0.9786562000000001, 0.9398936, 0.8847786, 0.8300944, 0.7422004, 0.6979599999999999, 0.6527299999999999, 0.5813058, 0.4815942, 0.3393904, 0.1714364, 0.1469552, 0.11168879999999999, 0.0987288, 0.0650324, 0.0589834],
        [0.9979825999999998, 0.9974413999999999, 0.9959703999999999, 0.9920699999999999, 0.9856180000000001, 0.9717798000000001, 0.950014, 0.8851516, 0.8288599999999999, 0.782005, 0.7193531999999999, 0.6765074, 0.6261798000000001, 0.5243462, 0.3981608, 0.262052, 0.18786740000000002, 0.12007480000000001, 0.0818204, 0.0798742, 0.067927],
        [0.9953108, 0.9975842, 0.9916226, 0.9917924, 0.9885290000000001, 0.9773671999999999, 0.958559, 0.895848, 0.8568058, 0.7818798000000001, 0.7286074, 0.6938409999999999, 0.6451194, 0.5644876, 0.4878194, 0.2673506, 0.18821060000000003, 0.1426718, 0.1055978, 0.109302, 0.0738052],
        [0.995034, 0.9963176000000001, 0.9923740000000001, 0.9942580000000001, 0.9879678, 0.9825002, 0.9703836, 0.9340524000000001, 0.8643114, 0.8108752, 0.746113, 0.7121268000000001, 0.6374572, 0.6016322000000001, 0.5183173999999999, 0.3633928, 0.23437360000000002, 0.16599060000000002, 0.1206306, 0.0951218, 0.0802436],
        [0.9963316, 0.9948480000000001, 0.9951327999999999, 0.9916412, 0.9923952, 0.9779566, 0.9793218000000001, 0.9483984000000001, 0.8992743999999999, 0.8180286, 0.7670220000000001, 0.7238348000000001, 0.6948716, 0.6590692000000001, 0.5918439999999999, 0.3639478, 0.2295712, 0.1860614, 0.1427998, 0.1165264, 0.10149699999999999],
        [0.9973874, 0.9944146, 0.9951888000000001, 0.9943492, 0.9931572, 0.9942369999999999, 0.9685262, 0.9411722, 0.8893676, 0.8486049999999998, 0.7937050000000001, 0.740198, 0.6999528, 0.6531834000000001, 0.6103141999999999, 0.4415552, 0.43194239999999995, 0.283855, 0.18557839999999998, 0.1163834, 0.1148766],
        [0.9966676, 0.9959198, 0.9968096, 0.9942526, 0.9916706, 0.9903979999999999, 0.9770905999999999, 0.9539465999999999, 0.9161092, 0.8823079999999999, 0.8149062, 0.7582032, 0.7108408, 0.6691935999999999, 0.6293164, 0.5608054, 0.5062132, 0.31880759999999997, 0.227594, 0.16797379999999998, 0.1363632],
        [0.9969714, 0.9942172000000002, 0.9945470000000001, 0.9941556, 0.9970320000000001, 0.9803548000000001, 0.9781125999999999, 0.9494865999999998, 0.9369626, 0.8864991999999999, 0.8467240000000001, 0.7921066, 0.7295023999999999, 0.7027182000000001, 0.6419108, 0.6143322, 0.5571694, 0.4400876, 0.3039978, 0.2071994, 0.12746680000000002],
        [0.9974086, 0.9972338, 0.9947326, 0.9931504000000001, 0.9897138000000001, 0.9897384000000001, 0.976596, 0.9588108, 0.9181398000000002, 0.908926, 0.8444078, 0.7925169999999998, 0.7558616, 0.715408, 0.6806902000000001, 0.6400579999999999, 0.5828018, 0.4480832, 0.42966139999999997, 0.2607666, 0.21684640000000002],
        [0.9989422, 0.9955766, 0.9977852000000001, 0.9945591999999999, 0.9918214000000001, 0.9934616, 0.9818452000000001, 0.9811964000000001, 0.9483018, 0.9136508000000001, 0.8668698, 0.8159444, 0.7735138, 0.7318812, 0.6878882000000001, 0.6708213999999999, 0.6144776000000001, 0.5642809999999999, 0.36474579999999995, 0.395091, 0.2503828],
        [0.9966035999999999, 0.996177, 0.9944118000000001, 0.9943868, 0.9950982, 0.985744, 0.9885652, 0.9765031999999999, 0.9565678, 0.9395018, 0.8702106000000001, 0.8626746, 0.796017, 0.7583590000000001, 0.7221502, 0.6808812000000001, 0.6644498, 0.6209076, 0.5269824, 0.4054012, 0.2726736],
        [0.9974466, 0.9954303999999999, 0.9962288000000001, 0.9907756000000001, 0.9916538, 0.989972, 0.9838271999999999, 0.9727726, 0.9745349999999999, 0.9579198, 0.8966014000000001, 0.8687788000000001, 0.8391348, 0.7858225999999999, 0.7407035999999999, 0.7215001999999999, 0.6736292, 0.6403488, 0.610308, 0.5303682, 0.4012196],
        [0.9988004, 0.9983895999999999, 0.9968458000000001, 0.9955905999999999, 0.9912378000000001, 0.9923684, 0.9897436000000001, 0.9871402, 0.9570756, 0.9414808000000001, 0.9014482000000001, 0.8734850000000001, 0.8423269999999999, 0.8101016000000001, 0.7572472, 0.7281773999999999, 0.7058196000000001, 0.6719628, 0.6300969999999999, 0.5716462, 0.5550736],
        [0.9992380, 0.99911289999, 0.9982323, 0.9989323, 0.9965232324, 0.997342,0.98974369999, 0.9871402, 0.9570756, 0.9414808000000001, 0.9014482000000001, 0.8734850000000001, 0.8623269999999999, 0.8201016000000001, 0.8072472, 0.7981773999999999, 0.7658196000000001, 0.6919628, 0.6799992300099999, 0.62000996462, 0.5999736],
]


m_values = np.arange(0.0, 1.1, 0.1)
b_values = np.arange(1.0, 2.1, 0.1)

X, Y = np.meshgrid(m_values, b_values)

matrix = np.array(f_3)

# 设置固定的图像大小
plt.figure(figsize=(8, 6))  # 这里可以根据需要调整宽度和高度
# 绘制热图，并将返回的图像对象保存到 imshow_obj
imshow_obj = plt.imshow(matrix, cmap='jet', interpolation='spline16', aspect='auto', extent=(0, 1, 1, 2))

cbar = plt.colorbar(imshow_obj)
cbar.ax.tick_params(labelsize=18)
cbar.set_label('$f_c$', fontsize=20, rotation=0)
cbar.ax.yaxis.set_label_position('right')  # 将label显示在右边
cbar.ax.yaxis.set_label_coords(3.5, 1)  # 设置label的位置，y轴值超过1将label上移


# 设置 imshow_obj 对象的颜色范围
imshow_obj.set_clim(0, 1)

plt.xlabel('m', fontsize=20)
plt.ylabel('b', fontsize=20,rotation=0,labelpad=10)
plt.xticks(m_values, fontsize=18)
plt.yticks(b_values, fontsize=18)
plt.grid(False)
plt.tight_layout()  # 这个命令能让图形的布局更紧凑，减少留白
plt.show()
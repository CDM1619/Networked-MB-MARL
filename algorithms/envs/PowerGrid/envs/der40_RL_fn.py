from algorithms.envs.PowerGrid.configs.parameters_der40 import *
import numba as nb


@nb.jit
def der_fn(x, t, disturbance_R, disturbance_L):
    # ---------Transferring Inv. Output Currents to Global DQ-----------------
    ioD1 = np.cos(0) * x[4] - np.sin(0) * x[5]
    ioQ1 = np.sin(0) * x[4] + np.cos(0) * x[5]
    ioD2 = np.cos(x[6]) * x[9] - np.sin(x[6]) * x[10]
    ioQ2 = np.sin(x[6]) * x[9] + np.cos(x[6]) * x[10]
    ioD3 = np.cos(x[11]) * x[14] - np.sin(x[11]) * x[15]
    ioQ3 = np.sin(x[11]) * x[14] + np.cos(x[11]) * x[15]
    ioD4 = np.cos(x[16]) * x[19] - np.sin(x[16]) * x[20]
    ioQ4 = np.sin(x[16]) * x[19] + np.cos(x[16]) * x[20]
    ioD5 = np.cos(x[21]) * x[24] - np.sin(x[21]) * x[25]
    ioQ5 = np.sin(x[21]) * x[24] + np.cos(x[21]) * x[25]
    ioD6 = np.cos(x[26]) * x[29] - np.sin(x[26]) * x[30]
    ioQ6 = np.sin(x[26]) * x[29] + np.cos(x[26]) * x[30]
    ioD7 = np.cos(x[31]) * x[34] - np.sin(x[31]) * x[35]
    ioQ7 = np.sin(x[31]) * x[34] + np.cos(x[31]) * x[35]
    ioD8 = np.cos(x[36]) * x[39] - np.sin(x[36]) * x[40]
    ioQ8 = np.sin(x[36]) * x[39] + np.cos(x[36]) * x[40]
    ioD9 = np.cos(x[41]) * x[44] - np.sin(x[41]) * x[45]
    ioQ9 = np.sin(x[41]) * x[44] + np.cos(x[41]) * x[45]
    ioD10 = np.cos(x[46]) * x[49] - np.sin(x[46]) * x[50]
    ioQ10 = np.sin(x[46]) * x[49] + np.cos(x[46]) * x[50]
    ioD11 = np.cos(x[51]) * x[54] - np.sin(x[51]) * x[55]
    ioQ11 = np.sin(x[51]) * x[54] + np.cos(x[51]) * x[55]
    ioD12 = np.cos(x[56]) * x[59] - np.sin(x[56]) * x[60]
    ioQ12 = np.sin(x[56]) * x[59] + np.cos(x[56]) * x[60]
    ioD13 = np.cos(x[61]) * x[64] - np.sin(x[61]) * x[65]
    ioQ13 = np.sin(x[61]) * x[64] + np.cos(x[61]) * x[65]
    ioD14 = np.cos(x[66]) * x[69] - np.sin(x[66]) * x[70]
    ioQ14 = np.sin(x[66]) * x[69] + np.cos(x[66]) * x[70]
    ioD15 = np.cos(x[71]) * x[74] - np.sin(x[71]) * x[75]
    ioQ15 = np.sin(x[71]) * x[74] + np.cos(x[71]) * x[75]
    ioD16 = np.cos(x[76]) * x[79] - np.sin(x[76]) * x[80]
    ioQ16 = np.sin(x[76]) * x[79] + np.cos(x[76]) * x[80]
    ioD17 = np.cos(x[81]) * x[84] - np.sin(x[81]) * x[85]
    ioQ17 = np.sin(x[81]) * x[84] + np.cos(x[81]) * x[85]
    ioD18 = np.cos(x[86]) * x[89] - np.sin(x[86]) * x[90]
    ioQ18 = np.sin(x[86]) * x[89] + np.cos(x[86]) * x[90]
    ioD19 = np.cos(x[91]) * x[94] - np.sin(x[91]) * x[95]
    ioQ19 = np.sin(x[91]) * x[94] + np.cos(x[91]) * x[95]
    ioD20 = np.cos(x[96]) * x[99] - np.sin(x[96]) * x[100]
    ioQ20 = np.sin(x[96]) * x[99] + np.cos(x[96]) * x[100]
    # 21 --> 30
    ioD21 = np.cos(101) * x[104] - np.sin(101) * x[105]
    ioQ21 = np.sin(101) * x[104] + np.cos(101) * x[105]
    ioD22 = np.cos(x[106]) * x[109] - np.sin(x[106]) * x[110]
    ioQ22 = np.sin(x[106]) * x[109] + np.cos(x[106]) * x[110]
    ioD23 = np.cos(x[111]) * x[114] - np.sin(x[111]) * x[115]
    ioQ23 = np.sin(x[111]) * x[114] + np.cos(x[111]) * x[115]
    ioD24 = np.cos(x[116]) * x[119] - np.sin(x[116]) * x[120]
    ioQ24 = np.sin(x[116]) * x[119] + np.cos(x[116]) * x[120]
    ioD25 = np.cos(x[121]) * x[124] - np.sin(x[121]) * x[125]
    ioQ25 = np.sin(x[121]) * x[124] + np.cos(x[121]) * x[125]
    ioD26 = np.cos(x[126]) * x[129] - np.sin(x[126]) * x[130]
    ioQ26 = np.sin(x[126]) * x[129] + np.cos(x[126]) * x[130]
    ioD27 = np.cos(x[131]) * x[134] - np.sin(x[131]) * x[135]
    ioQ27 = np.sin(x[131]) * x[134] + np.cos(x[131]) * x[135]
    ioD28 = np.cos(x[136]) * x[139] - np.sin(x[136]) * x[140]
    ioQ28 = np.sin(x[136]) * x[139] + np.cos(x[136]) * x[140]
    ioD29 = np.cos(x[141]) * x[144] - np.sin(x[141]) * x[145]
    ioQ29 = np.sin(x[141]) * x[144] + np.cos(x[141]) * x[145]
    ioD30 = np.cos(x[146]) * x[149] - np.sin(x[146]) * x[150]
    ioQ30 = np.sin(x[146]) * x[149] + np.cos(x[146]) * x[150]
    # 31 --> 40
    ioD31 = np.cos(x[151]) * x[154] - np.sin(x[151]) * x[155]
    ioQ31 = np.sin(x[151]) * x[154] + np.cos(x[151]) * x[155]
    ioD32 = np.cos(x[156]) * x[159] - np.sin(x[156]) * x[160]
    ioQ32 = np.sin(x[156]) * x[159] + np.cos(x[156]) * x[160]
    ioD33 = np.cos(x[161]) * x[164] - np.sin(x[161]) * x[165]
    ioQ33 = np.sin(x[161]) * x[164] + np.cos(x[161]) * x[165]
    ioD34 = np.cos(x[166]) * x[169] - np.sin(x[166]) * x[170]
    ioQ34 = np.sin(x[166]) * x[169] + np.cos(x[166]) * x[170]
    ioD35 = np.cos(x[171]) * x[174] - np.sin(x[171]) * x[175]
    ioQ35 = np.sin(x[171]) * x[174] + np.cos(x[171]) * x[175]
    ioD36 = np.cos(x[176]) * x[179] - np.sin(x[176]) * x[180]
    ioQ36 = np.sin(x[176]) * x[179] + np.cos(x[176]) * x[180]
    ioD37 = np.cos(x[181]) * x[184] - np.sin(x[181]) * x[185]
    ioQ37 = np.sin(x[181]) * x[184] + np.cos(x[181]) * x[185]
    ioD38 = np.cos(x[186]) * x[189] - np.sin(x[186]) * x[190]
    ioQ38 = np.sin(x[186]) * x[189] + np.cos(x[186]) * x[190]
    ioD39 = np.cos(x[191]) * x[194] - np.sin(x[191]) * x[195]
    ioQ39 = np.sin(x[191]) * x[194] + np.cos(x[191]) * x[195]
    ioD40 = np.cos(x[196]) * x[199] - np.sin(x[196]) * x[200]
    ioQ40 = np.sin(x[196]) * x[199] + np.cos(x[196]) * x[200]

    # ---------Defining Bus Voltages-----------------
    vbD1 = rN * (ioD1 - x[285] - x[201])
    vbQ1 = rN * (ioQ1 - x[286] - x[202])
    vbD2 = rN * (ioD2 + x[205] - x[207])
    vbQ2 = rN * (ioQ2 + x[206] - x[208])
    vbD3 = rN * (ioD3 + x[207] - x[209] - x[289])
    vbQ3 = rN * (ioQ3 + x[208] - x[210] - x[290])
    vbD4 = rN * (ioD4 + x[213] - x[215])
    vbQ4 = rN * (ioQ4 + x[214] - x[216])
    vbD5 = rN * (ioD5 + x[215] - x[217] - x[291])
    vbQ5 = rN * (ioQ5 + x[216] - x[218] - x[292])
    vbD6 = rN * (ioD6 + x[201] - x[203])
    vbQ6 = rN * (ioQ6 + x[202] - x[204])
    vbD7 = rN * (ioD7 + x[203] - x[205] - x[287])
    vbQ7 = rN * (ioQ7 + x[204] - x[206] - x[288])
    vbD8 = rN * (ioD8 + x[209] - x[211] - x[219])
    vbQ8 = rN * (ioQ8 + x[210] - x[212] - x[220])
    vbD9 = rN * (ioD9 + x[211] - x[213] - x[291])
    vbQ9 = rN * (ioQ9 + x[212] - x[214] - x[292])
    vbD10 = rN * (ioD10 + x[217] - x[221])
    vbQ10 = rN * (ioQ10 + x[218] - x[222])
    vbD11 = rN * (ioD11 + x[219] - x[223] - x[295])
    vbQ11 = rN * (ioQ11 + x[220] - x[224] - x[296])
    vbD12 = rN * (ioD12 + x[227] - x[229])
    vbQ12 = rN * (ioQ12 + x[228] - x[230])
    vbD13 = rN * (ioD13 + x[221] + x[229] - x[231] - x[299])
    vbQ13 = rN * (ioQ13 + x[222] + x[230] - x[232] - x[300])
    vbD14 = rN * (ioD14 + x[235] - x[237])
    vbQ14 = rN * (ioQ14 + x[236] - x[238])
    vbD15 = rN * (ioD15 + x[237] - x[239] - x[303])
    vbQ15 = rN * (ioQ15 + x[238] - x[240] - x[304])
    vbD16 = rN * (ioD16 + x[223] - x[225])
    vbQ16 = rN * (ioQ16 + x[224] - x[226])
    vbD17 = rN * (ioD17 + x[225] - x[227] - x[297])
    vbQ17 = rN * (ioQ17 + x[226] - x[228] - x[298])
    vbD18 = rN * (ioD18 + x[231] - x[233] - x[281])
    vbQ18 = rN * (ioQ18 + x[232] - x[234] - x[282])
    vbD19 = rN * (ioD19 + x[233] - x[235] - x[301])
    vbQ19 = rN * (ioQ19 + x[234] - x[236] - x[302])
    vbD20 = rN * (ioD20 + x[239] - x[283])
    vbQ20 = rN * (ioQ20 + x[240] - x[284])
    # 21 --> 30
    vbD21 = rN * (ioD21 + x[281] - x[305] - x[241])
    vbQ21 = rN * (ioQ21 + x[282] - x[306] - x[242])
    vbD22 = rN * (ioD22 + x[245] - x[247])
    vbQ22 = rN * (ioQ22 + x[246] - x[248])
    vbD23 = rN * (ioD23 + x[247] + x[283] - x[249] - x[309])
    vbQ23 = rN * (ioQ23 + x[248] + x[284] - x[250] - x[310])
    vbD24 = rN * (ioD24 + x[253] - x[255])
    vbQ24 = rN * (ioQ24 + x[254] - x[256])
    vbD25 = rN * (ioD25 + x[255] - x[257] - x[311])
    vbQ25 = rN * (ioQ25 + x[256] - x[258] - x[312])
    vbD26 = rN * (ioD26 + x[241] - x[243])
    vbQ26 = rN * (ioQ26 + x[242] - x[244])
    vbD27 = rN * (ioD27 + x[243] - x[245] - x[307])
    vbQ27 = rN * (ioQ27 + x[244] - x[246] - x[308])
    vbD28 = rN * (ioD28 + x[249] - x[251] - x[259])
    vbQ28 = rN * (ioQ28 + x[250] - x[252] - x[260])
    vbD29 = rN * (ioD29 + x[251] - x[253] - x[311])
    vbQ29 = rN * (ioQ29 + x[252] - x[254] - x[312])
    vbD30 = rN * (ioD30 + x[257] - x[261])
    vbQ30 = rN * (ioQ30 + x[258] - x[262])
    # 31 --> 40
    vbD31 = rN * (ioD31 + x[259] - x[263] - x[315])
    vbQ31 = rN * (ioQ31 + x[260] - x[264] - x[316])
    vbD32 = rN * (ioD32 + x[267] - x[269])
    vbQ32 = rN * (ioQ32 + x[268] - x[270])
    vbD33 = rN * (ioD33 + x[261] + x[269] - x[271] - x[319])
    vbQ33 = rN * (ioQ33 + x[262] + x[270] - x[272] - x[320])
    vbD34 = rN * (ioD34 + x[275] - x[277])
    vbQ34 = rN * (ioQ34 + x[276] - x[278])
    vbD35 = rN * (ioD35 + x[277] - x[279] - x[323])
    vbQ35 = rN * (ioQ35 + x[278] - x[280] - x[324])
    vbD36 = rN * (ioD36 + x[263] - x[265])
    vbQ36 = rN * (ioQ36 + x[264] - x[266])
    vbD37 = rN * (ioD37 + x[265] - x[267] - x[317])
    vbQ37 = rN * (ioQ37 + x[266] - x[268] - x[318])
    vbD38 = rN * (ioD38 + x[271] - x[273])
    vbQ38 = rN * (ioQ38 + x[272] - x[274])
    vbD39 = rN * (ioD39 + x[273] - x[275] - x[321])
    vbQ39 = rN * (ioQ39 + x[274] - x[276] - x[322])
    vbD40 = rN * (ioD40 + x[279])
    vbQ40 = rN * (ioQ40 + x[280])

    # ---------Transferring Bus Voltages to Inv. dq-----------------
    vbd1 = np.cos(0) * vbD1 + np.sin(0) * vbQ1
    vbq1 = -np.sin(0) * vbD1 + np.cos(0) * vbQ1
    vbd2 = np.cos(x[6]) * vbD22 + np.sin(x[6]) * vbQ2
    vbq2 = -np.sin(x[6]) * vbD2 + np.cos(x[6]) * vbQ2
    vbd3 = np.cos(x[11]) * vbD3 + np.sin(x[11]) * vbQ3
    vbq3 = -np.sin(x[11]) * vbD3 + np.cos(x[11]) * vbQ3
    vbd4 = np.cos(x[16]) * vbD4 + np.sin(x[16]) * vbQ4
    vbq4 = -np.sin(x[16]) * vbD4 + np.cos(x[16]) * vbQ4
    vbd5 = np.cos(x[21]) * vbD5 + np.sin(x[21]) * vbQ5
    vbq5 = -np.sin(x[21]) * vbD5 + np.cos(x[21]) * vbQ5
    vbd6 = np.cos(x[26]) * vbD6 + np.sin(x[26]) * vbQ6
    vbq6 = -np.sin(x[26]) * vbD6 + np.cos(x[26]) * vbQ6
    vbd7 = np.cos(x[31]) * vbD7 + np.sin(x[31]) * vbQ7
    vbq7 = -np.sin(x[31]) * vbD7 + np.cos(x[31]) * vbQ7
    vbd8 = np.cos(x[36]) * vbD8 + np.sin(x[36]) * vbQ8
    vbq8 = -np.sin(x[36]) * vbD8 + np.cos(x[36]) * vbQ8
    vbd9 = np.cos(x[41]) * vbD9 + np.sin(x[41]) * vbQ9
    vbq9 = -np.sin(x[41]) * vbD9 + np.cos(x[41]) * vbQ9
    vbd10 = np.cos(x[46]) * vbD10 + np.sin(x[46]) * vbQ10
    vbq10 = -np.sin(x[46]) * vbD10 + np.cos(x[46]) * vbQ10
    vbd11 = np.cos(x[51]) * vbD11 + np.sin(x[51]) * vbQ11
    vbq11 = -np.sin(x[51]) * vbD11 + np.cos(x[51]) * vbQ11
    vbd12 = np.cos(x[56]) * vbD12 + np.sin(x[56]) * vbQ12
    vbq12 = -np.sin(x[56]) * vbD12 + np.cos(x[56]) * vbQ12
    vbd13 = np.cos(x[61]) * vbD13 + np.sin(x[61]) * vbQ13
    vbq13 = -np.sin(x[61]) * vbD13 + np.cos(x[61]) * vbQ13
    vbd14 = np.cos(x[66]) * vbD14 + np.sin(x[66]) * vbQ14
    vbq14 = -np.sin(x[66]) * vbD14 + np.cos(x[66]) * vbQ14
    vbd15 = np.cos(x[71]) * vbD15 + np.sin(x[71]) * vbQ15
    vbq15 = -np.sin(x[71]) * vbD15 + np.cos(x[71]) * vbQ15
    vbd16 = np.cos(x[76]) * vbD16 + np.sin(x[76]) * vbQ16
    vbq16 = -np.sin(x[76]) * vbD16 + np.cos(x[76]) * vbQ16
    vbd17 = np.cos(x[81]) * vbD17 + np.sin(x[81]) * vbQ17
    vbq17 = -np.sin(x[81]) * vbD17 + np.cos(x[81]) * vbQ17
    vbd18 = np.cos(x[86]) * vbD18 + np.sin(x[86]) * vbQ18
    vbq18 = -np.sin(x[86]) * vbD18 + np.cos(x[86]) * vbQ18
    vbd19 = np.cos(x[91]) * vbD19 + np.sin(x[91]) * vbQ19
    vbq19 = -np.sin(x[91]) * vbD19 + np.cos(x[91]) * vbQ19
    vbd20 = np.cos(x[96]) * vbD20 + np.sin(x[96]) * vbQ20
    vbq20 = -np.sin(x[96]) * vbD20 + np.cos(x[96]) * vbQ20
    # 21 --> 30
    vbd21 = np.cos(101) * vbD21 + np.sin(101) * vbQ21
    vbq21 = -np.sin(101) * vbD21 + np.cos(101) * vbQ21
    vbd22 = np.cos(x[106]) * vbD22 + np.sin(x[106]) * vbQ22
    vbq22 = -np.sin(x[106]) * vbD22 + np.cos(x[106]) * vbQ22
    vbd23 = np.cos(x[111]) * vbD23 + np.sin(x[111]) * vbQ23
    vbq23 = -np.sin(x[111]) * vbD23 + np.cos(x[111]) * vbQ23
    vbd24 = np.cos(x[116]) * vbD24 + np.sin(x[116]) * vbQ24
    vbq24 = -np.sin(x[116]) * vbD24 + np.cos(x[116]) * vbQ24
    vbd25 = np.cos(x[121]) * vbD25 + np.sin(x[121]) * vbQ25
    vbq25 = -np.sin(x[121]) * vbD25 + np.cos(x[121]) * vbQ25
    vbd26 = np.cos(x[126]) * vbD26 + np.sin(x[126]) * vbQ26
    vbq26 = -np.sin(x[126]) * vbD26 + np.cos(x[126]) * vbQ26
    vbd27 = np.cos(x[131]) * vbD27 + np.sin(x[131]) * vbQ27
    vbq27 = -np.sin(x[131]) * vbD27 + np.cos(x[131]) * vbQ27
    vbd28 = np.cos(x[136]) * vbD28 + np.sin(x[136]) * vbQ28
    vbq28 = -np.sin(x[136]) * vbD28 + np.cos(x[136]) * vbQ28
    vbd29 = np.cos(x[141]) * vbD29 + np.sin(x[141]) * vbQ29
    vbq29 = -np.sin(x[141]) * vbD29 + np.cos(x[141]) * vbQ29
    vbd30 = np.cos(x[146]) * vbD30 + np.sin(x[146]) * vbQ30
    vbq30 = -np.sin(x[146]) * vbD30 + np.cos(x[146]) * vbQ30
    # 31 --> 40
    vbd31 = np.cos(x[151]) * vbD31 + np.sin(x[151]) * vbQ31
    vbq31 = -np.sin(x[151]) * vbD31 + np.cos(x[151]) * vbQ31
    vbd32 = np.cos(x[156]) * vbD32 + np.sin(x[156]) * vbQ32
    vbq32 = -np.sin(x[156]) * vbD32 + np.cos(x[156]) * vbQ32
    vbd33 = np.cos(x[161]) * vbD33 + np.sin(x[161]) * vbQ33
    vbq33 = -np.sin(x[161]) * vbD33 + np.cos(x[161]) * vbQ33
    vbd34 = np.cos(x[166]) * vbD34 + np.sin(x[166]) * vbQ34
    vbq34 = -np.sin(x[166]) * vbD34 + np.cos(x[166]) * vbQ34
    vbd35 = np.cos(x[171]) * vbD35 + np.sin(x[171]) * vbQ35
    vbq35 = -np.sin(x[171]) * vbD35 + np.cos(x[171]) * vbQ35
    vbd36 = np.cos(x[176]) * vbD36 + np.sin(x[176]) * vbQ36
    vbq36 = -np.sin(x[176]) * vbD36 + np.cos(x[176]) * vbQ36
    vbd37 = np.cos(x[181]) * vbD37 + np.sin(x[181]) * vbQ37
    vbq37 = -np.sin(x[181]) * vbD37 + np.cos(x[181]) * vbQ37
    vbd38 = np.cos(x[186]) * vbD38 + np.sin(x[186]) * vbQ38
    vbq38 = -np.sin(x[186]) * vbD38 + np.cos(x[186]) * vbQ38
    vbd39 = np.cos(x[191]) * vbD39 + np.sin(x[191]) * vbQ39
    vbq39 = -np.sin(x[191]) * vbD39 + np.cos(x[191]) * vbQ39
    vbd40 = np.cos(x[196]) * vbD40 + np.sin(x[196]) * vbQ40
    vbq40 = -np.sin(x[196]) * vbD40 + np.cos(x[196]) * vbQ40

    wcom = x[325] - mp1 * x[2]
    # ---------------DG1--------------------
    vod1_star = x[365] - nq1 * x[3]
    voq1_star = 0
    xdot1 = x[325] - mp1 * x[2] - wcom
    xdot2 = wc * (vod1_star * x[4] + voq1_star * x[5] - x[2])
    xdot3 = wc * (-vod1_star * x[5] + voq1_star * x[4] - x[3])
    xdot4 = (-rLc / Lc) * x[4] + wcom * x[5] + (1 / Lc) * (vod1_star - vbd1)
    xdot5 = (-rLc / Lc) * x[5] - wcom * x[4] + (1 / Lc) * (voq1_star - vbq1)

    # ----------------DG2-------------------
    vod2_star = x[366] - nq2 * x[8]
    voq2_star = 0
    xdot6 = x[326] - mp2 * x[7] - wcom
    xdot7 = wc * (vod2_star * x[9] + voq2_star * x[10] - x[7])
    xdot8 = wc * (-vod2_star * x[10] + voq2_star * x[9] - x[8])
    xdot9 = (-rLc / Lc) * x[9] + wcom * x[10] + (1 / Lc) * (vod2_star - vbd2)
    xdot10 = (-rLc / Lc) * x[10] - wcom * x[9] + (1 / Lc) * (voq2_star - vbq2)

    # ----------------DG3-------------------
    vod3_star = x[367] - nq3 * x[13]
    voq3_star = 0
    xdot11 = x[327] - mp3 * x[12] - wcom
    xdot12 = wc * (vod3_star * x[14] + voq3_star * x[15] - x[12])
    xdot13 = wc * (-vod3_star * x[15] + voq3_star * x[14] - x[13])
    xdot14 = (-rLc / Lc) * x[14] + wcom * x[15] + (1 / Lc) * (vod3_star - vbd3)
    xdot15 = (-rLc / Lc) * x[15] - wcom * x[14] + (1 / Lc) * (voq3_star - vbq3)

    # ----------------DG4-------------------
    vod4_star = x[368] - nq4 * x[18]
    voq4_star = 0
    xdot16 = x[328] - mp4 * x[17] - wcom
    xdot17 = wc * (vod4_star * x[19] + voq4_star * x[20] - x[17])
    xdot18 = wc * (-vod4_star * x[20] + voq4_star * x[19] - x[18])
    xdot19 = (-rLc / Lc) * x[19] + wcom * x[20] + (1 / Lc) * (vod4_star - vbd4)
    xdot20 = (-rLc / Lc) * x[20] - wcom * x[19] + (1 / Lc) * (voq4_star - vbq4)

    # ----------------DG5-------------------
    vod5_star = x[369] - nq5 * x[23]
    voq5_star = 0
    xdot21 = x[329] - mp5 * x[22] - wcom
    xdot22 = wc * (vod5_star * x[24] + voq5_star * x[25] - x[22])
    xdot23 = wc * (-vod5_star * x[25] + voq5_star * x[24] - x[23])
    xdot24 = (-rLc / Lc) * x[24] + wcom * x[25] + (1 / Lc) * (vod5_star - vbd5)
    xdot25 = (-rLc / Lc) * x[25] - wcom * x[24] + (1 / Lc) * (voq5_star - vbq5)

    # ----------------DG6-------------------
    vod6_star = x[370] - nq6 * x[28]
    voq6_star = 0
    xdot26 = x[330] - mp6 * x[27] - wcom
    xdot27 = wc * (vod6_star * x[29] + voq6_star * x[30] - x[27])
    xdot28 = wc * (-vod6_star * x[30] + voq6_star * x[29] - x[28])
    xdot29 = (-rLc / Lc) * x[29] + wcom * x[30] + (1 / Lc) * (vod6_star - vbd6)
    xdot30 = (-rLc / Lc) * x[30] - wcom * x[29] + (1 / Lc) * (voq6_star - vbq6)

    # ----------------DG7-------------------
    vod7_star = x[371] - nq7 * x[33]
    voq7_star = 0
    xdot31 = x[331] - mp7 * x[32] - wcom
    xdot32 = wc * (vod7_star * x[34] + voq7_star * x[35] - x[32])
    xdot33 = wc * (-vod7_star * x[35] + voq7_star * x[34] - x[33])
    xdot34 = (-rLc / Lc) * x[34] + wcom * x[35] + (1 / Lc) * (vod7_star - vbd7)
    xdot35 = (-rLc / Lc) * x[35] - wcom * x[34] + (1 / Lc) * (voq7_star - vbq7)

    # ----------------DG8-------------------
    vod8_star = x[372] - nq8 * x[38]
    voq8_star = 0
    xdot36 = x[332] - mp8 * x[37] - wcom
    xdot37 = wc * (vod8_star * x[39] + voq8_star * x[40] - x[37])
    xdot38 = wc * (-vod8_star * x[40] + voq8_star * x[39] - x[38])
    xdot39 = (-rLc / Lc) * x[39] + wcom * x[40] + (1 / Lc) * (vod8_star - vbd8)
    xdot40 = (-rLc / Lc) * x[40] - wcom * x[39] + (1 / Lc) * (voq8_star - vbq8)

    # ----------------DG9-------------------
    vod9_star = x[373] - nq9 * x[43]
    voq9_star = 0
    xdot41 = x[333] - mp9 * x[42] - wcom
    xdot42 = wc * (vod9_star * x[44] + voq9_star * x[45] - x[42])
    xdot43 = wc * (-vod9_star * x[45] + voq9_star * x[44] - x[43])
    xdot44 = (-rLc / Lc) * x[44] + wcom * x[45] + (1 / Lc) * (vod9_star - vbd9)
    xdot45 = (-rLc / Lc) * x[45] - wcom * x[44] + (1 / Lc) * (voq9_star - vbq9)

    # ----------------DG10-------------------
    vod10_star = x[374] - nq10 * x[48]
    voq10_star = 0
    xdot46 = x[334] - mp10 * x[47] - wcom
    xdot47 = wc * (vod10_star * x[49] + voq10_star * x[50] - x[47])
    xdot48 = wc * (-vod10_star * x[50] + voq10_star * x[49] - x[48])
    xdot49 = (-rLc / Lc) * x[49] + wcom * x[50] + (1 / Lc) * (vod10_star - vbd10)
    xdot50 = (-rLc / Lc) * x[50] - wcom * x[49] + (1 / Lc) * (voq10_star - vbq10)

    # ----------------DG11-------------------
    vod11_star = x[375] - nq11 * x[53]
    voq11_star = 0
    xdot51 = x[335] - mp11 * x[52] - wcom
    xdot52 = wc * (vod11_star * x[54] + voq11_star * x[55] - x[52])
    xdot53 = wc * (-vod11_star * x[55] + voq11_star * x[54] - x[53])
    xdot54 = (-rLc / Lc) * x[54] + wcom * x[55] + (1 / Lc) * (vod11_star - vbd11)
    xdot55 = (-rLc / Lc) * x[55] - wcom * x[54] + (1 / Lc) * (voq11_star - vbq11)

    # ----------------DG12-------------------
    vod12_star = x[376] - nq12 * x[58]
    voq12_star = 0
    xdot56 = x[336] - mp12 * x[57] - wcom
    xdot57 = wc * (vod12_star * x[59] + voq12_star * x[60] - x[57])
    xdot58 = wc * (-vod12_star * x[60] + voq12_star * x[59] - x[58])
    xdot59 = (-rLc / Lc) * x[59] + wcom * x[60] + (1 / Lc) * (vod12_star - vbd12)
    xdot60 = (-rLc / Lc) * x[60] - wcom * x[59] + (1 / Lc) * (voq12_star - vbq12)

    # ----------------DG13-------------------
    vod13_star = x[377] - nq13 * x[63]
    voq13_star = 0
    xdot61 = x[337] - mp13 * x[62] - wcom
    xdot62 = wc * (vod13_star * x[64] + voq13_star * x[65] - x[62])
    xdot63 = wc * (-vod13_star * x[65] + voq13_star * x[54] - x[63])
    xdot64 = (-rLc / Lc) * x[64] + wcom * x[65] + (1 / Lc) * (vod13_star - vbd13)
    xdot65 = (-rLc / Lc) * x[65] - wcom * x[64] + (1 / Lc) * (voq13_star - vbq13)

    # ----------------DG14-------------------
    vod14_star = x[378] - nq14 * x[68]
    voq14_star = 0
    xdot66 = x[338] - mp14 * x[67] - wcom
    xdot67 = wc * (vod14_star * x[69] + voq14_star * x[70] - x[67])
    xdot68 = wc * (-vod14_star * x[70] + voq14_star * x[69] - x[68])
    xdot69 = (-rLc / Lc) * x[69] + wcom * x[70] + (1 / Lc) * (vod14_star - vbd14)
    xdot70 = (-rLc / Lc) * x[70] - wcom * x[69] + (1 / Lc) * (voq14_star - vbq14)

    # ----------------DG15-------------------
    vod15_star = x[379] - nq15 * x[73]
    voq15_star = 0
    xdot71 = x[339] - mp15 * x[72] - wcom
    xdot72 = wc * (vod15_star * x[74] + voq15_star * x[75] - x[72])
    xdot73 = wc * (-vod15_star * x[75] + voq15_star * x[74] - x[73])
    xdot74 = (-rLc / Lc) * x[74] + wcom * x[75] + (1 / Lc) * (vod15_star - vbd15)
    xdot75 = (-rLc / Lc) * x[75] - wcom * x[74] + (1 / Lc) * (voq15_star - vbq15)

    # ----------------DG16-------------------
    vod16_star = x[380] - nq16 * x[78]
    voq16_star = 0
    xdot76 = x[340] - mp16 * x[77] - wcom
    xdot77 = wc * (vod16_star * x[79] + voq16_star * x[80] - x[77])
    xdot78 = wc * (-vod16_star * x[80] + voq16_star * x[79] - x[78])
    xdot79 = (-rLc / Lc) * x[79] + wcom * x[80] + (1 / Lc) * (vod16_star - vbd16)
    xdot80 = (-rLc / Lc) * x[80] - wcom * x[79] + (1 / Lc) * (voq16_star - vbq16)

    # ----------------DG17-------------------
    vod17_star = x[381] - nq17 * x[83]
    voq17_star = 0
    xdot81 = x[341] - mp17 * x[82] - wcom
    xdot82 = wc * (vod17_star * x[84] + voq17_star * x[85] - x[82])
    xdot83 = wc * (-vod17_star * x[85] + voq17_star * x[84] - x[83])
    xdot84 = (-rLc / Lc) * x[84] + wcom * x[85] + (1 / Lc) * (vod17_star - vbd17)
    xdot85 = (-rLc / Lc) * x[85] - wcom * x[84] + (1 / Lc) * (voq17_star - vbq17)

    # ----------------DG18-------------------
    vod18_star = x[382] - nq18 * x[88]
    voq18_star = 0
    xdot86 = x[342] - mp18 * x[87] - wcom
    xdot87 = wc * (vod18_star * x[89] + voq18_star * x[90] - x[87])
    xdot88 = wc * (-vod18_star * x[90] + voq18_star * x[89] - x[88])
    xdot89 = (-rLc / Lc) * x[89] + wcom * x[90] + (1 / Lc) * (vod18_star - vbd18)
    xdot90 = (-rLc / Lc) * x[90] - wcom * x[89] + (1 / Lc) * (voq18_star - vbq18)

    # ----------------DG19-------------------
    vod19_star = x[383] - nq19 * x[93]
    voq19_star = 0
    xdot91 = x[343] - mp19 * x[92] - wcom
    xdot92 = wc * (vod19_star * x[94] + voq19_star * x[95] - x[92])
    xdot93 = wc * (-vod19_star * x[95] + voq19_star * x[94] - x[93])
    xdot94 = (-rLc / Lc) * x[94] + wcom * x[95] + (1 / Lc) * (vod19_star - vbd19)
    xdot95 = (-rLc / Lc) * x[95] - wcom * x[94] + (1 / Lc) * (voq19_star - vbq19)

    # ----------------DG20-------------------
    vod20_star = x[384] - nq20 * x[98]
    voq20_star = 0
    xdot96 = x[344] - mp20 * x[97] - wcom
    xdot97 = wc * (vod20_star * x[99] + voq20_star * x[100] - x[97])
    xdot98 = wc * (-vod20_star * x[100] + voq20_star * x[99] - x[98])
    xdot99 = (-rLc / Lc) * x[99] + wcom * x[100] + (1 / Lc) * (vod20_star - vbd20)
    xdot100 = (-rLc / Lc) * x[100] - wcom * x[99] + (1 / Lc) * (voq20_star - vbq20)

    # ----------------DG21-------------------
    vod21_star = x[385] - nq21 * x[103]
    voq21_star = 0
    xdot101 = x[345] - mp21 * x[102] - wcom
    xdot102 = wc * (vod21_star * x[104] + voq21_star * x[105] - x[102])
    xdot103 = wc * (-vod21_star * x[105] + voq21_star * x[104] - x[103])
    xdot104 = (-rLc / Lc) * x[104] + wcom * x[105] + (1 / Lc) * (vod21_star - vbd21)
    xdot105 = (-rLc / Lc) * x[105] - wcom * x[104] + (1 / Lc) * (voq21_star - vbq21)

    # ----------------DG22-------------------
    vod22_star = x[386] - nq22 * x[108]
    voq22_star = 0
    xdot106 = x[346] - mp22 * x[107] - wcom
    xdot107 = wc * (vod22_star * x[109] + voq22_star * x[110] - x[107])
    xdot108 = wc * (-vod22_star * x[110] + voq22_star * x[109] - x[108])
    xdot109 = (-rLc / Lc) * x[109] + wcom * x[110] + (1 / Lc) * (vod22_star - vbd22)
    xdot110 = (-rLc / Lc) * x[110] - wcom * x[109] + (1 / Lc) * (voq22_star - vbq22)

    # ----------------DG23-------------------
    vod23_star = x[387] - nq23 * x[113]
    voq23_star = 0
    xdot111 = x[347] - mp23 * x[112] - wcom
    xdot112 = wc * (vod23_star * x[114] + voq23_star * x[115] - x[112])
    xdot113 = wc * (-vod23_star * x[115] + voq23_star * x[114] - x[113])
    xdot114 = (-rLc / Lc) * x[114] + wcom * x[115] + (1 / Lc) * (vod23_star - vbd23)
    xdot115 = (-rLc / Lc) * x[115] - wcom * x[114] + (1 / Lc) * (voq23_star - vbq23)

    # ----------------DG24------------------
    vod24_star = x[388] - nq24 * x[118]
    voq24_star = 0
    xdot116 = x[348] - mp24 * x[117] - wcom
    xdot117 = wc * (vod24_star * x[119] + voq24_star * x[120] - x[117])
    xdot118 = wc * (-vod24_star * x[120] + voq24_star * x[119] - x[118])
    xdot119 = (-rLc / Lc) * x[119] + wcom * x[120] + (1 / Lc) * (vod24_star - vbd24)
    xdot120 = (-rLc / Lc) * x[120] - wcom * x[119] + (1 / Lc) * (voq24_star - vbq24)

    # ----------------DG25-------------------
    vod25_star = x[389] - nq25 * x[123]
    voq25_star = 0
    xdot121 = x[349] - mp25 * x[122] - wcom
    xdot122 = wc * (vod25_star * x[124] + voq25_star * x[125] - x[122])
    xdot123 = wc * (-vod25_star * x[125] + voq25_star * x[124] - x[123])
    xdot124 = (-rLc / Lc) * x[124] + wcom * x[125] + (1 / Lc) * (vod25_star - vbd25)
    xdot125 = (-rLc / Lc) * x[125] - wcom * x[124] + (1 / Lc) * (voq25_star - vbq25)

    # ----------------DG26-------------------
    vod26_star = x[390] - nq26 * x[128]
    voq26_star = 0
    xdot126 = x[350] - mp26 * x[127] - wcom
    xdot127 = wc * (vod26_star * x[129] + voq26_star * x[130] - x[127])
    xdot128 = wc * (-vod26_star * x[130] + voq26_star * x[129] - x[128])
    xdot129 = (-rLc / Lc) * x[129] + wcom * x[130] + (1 / Lc) * (vod26_star - vbd26)
    xdot130 = (-rLc / Lc) * x[130] - wcom * x[129] + (1 / Lc) * (voq26_star - vbq26)

    # ----------------DG27-------------------
    vod27_star = x[391] - nq27 * x[133]
    voq27_star = 0
    xdot131 = x[351] - mp27 * x[132] - wcom
    xdot132 = wc * (vod27_star * x[134] + voq27_star * x[135] - x[132])
    xdot133 = wc * (-vod27_star * x[135] + voq27_star * x[134] - x[133])
    xdot134 = (-rLc / Lc) * x[134] + wcom * x[135] + (1 / Lc) * (vod27_star - vbd27)
    xdot135 = (-rLc / Lc) * x[135] - wcom * x[134] + (1 / Lc) * (voq27_star - vbq27)

    # ----------------DG28------------------
    vod28_star = x[392] - nq28 * x[138]
    voq28_star = 0
    xdot136 = x[352] - mp28 * x[137] - wcom
    xdot137 = wc * (vod28_star * x[139] + voq28_star * x[140] - x[137])
    xdot138 = wc * (-vod28_star * x[140] + voq28_star * x[139] - x[138])
    xdot139 = (-rLc / Lc) * x[139] + wcom * x[140] + (1 / Lc) * (vod28_star - vbd28)
    xdot140 = (-rLc / Lc) * x[140] - wcom * x[139] + (1 / Lc) * (voq28_star - vbq28)

    # ----------------DG29-------------------
    vod29_star = x[393] - nq29 * x[143]
    voq29_star = 0
    xdot141 = x[353] - mp29 * x[142] - wcom
    xdot142 = wc * (vod29_star * x[144] + voq29_star * x[145] - x[142])
    xdot143 = wc * (-vod29_star * x[145] + voq29_star * x[144] - x[143])
    xdot144 = (-rLc / Lc) * x[144] + wcom * x[145] + (1 / Lc) * (vod29_star - vbd29)
    xdot145 = (-rLc / Lc) * x[145] - wcom * x[144] + (1 / Lc) * (voq29_star - vbq29)

    # ----------------DG30-------------------
    vod30_star = x[394] - nq30 * x[148]
    voq30_star = 0
    xdot146 = x[354] - mp30 * x[147] - wcom
    xdot147 = wc * (vod30_star * x[149] + voq30_star * x[150] - x[147])
    xdot148 = wc * (-vod30_star * x[150] + voq30_star * x[149] - x[148])
    xdot149 = (-rLc / Lc) * x[149] + wcom * x[150] + (1 / Lc) * (vod30_star - vbd30)
    xdot150 = (-rLc / Lc) * x[150] - wcom * x[149] + (1 / Lc) * (voq30_star - vbq30)

    # ----------------DG31-------------------
    vod31_star = x[395] - nq31 * x[153]
    voq31_star = 0
    xdot151 = x[355] - mp31 * x[152] - wcom
    xdot152 = wc * (vod31_star * x[154] + voq31_star * x[155] - x[152])
    xdot153 = wc * (-vod31_star * x[155] + voq31_star * x[154] - x[153])
    xdot154 = (-rLc / Lc) * x[154] + wcom * x[155] + (1 / Lc) * (vod31_star - vbd31)
    xdot155 = (-rLc / Lc) * x[155] - wcom * x[154] + (1 / Lc) * (voq31_star - vbq31)

    # ----------------DG32-------------------
    vod32_star = x[396] - nq32 * x[158]
    voq32_star = 0
    xdot156 = x[356] - mp32 * x[157] - wcom
    xdot157 = wc * (vod32_star * x[159] + voq32_star * x[160] - x[157])
    xdot158 = wc * (-vod32_star * x[160] + voq32_star * x[159] - x[158])
    xdot159 = (-rLc / Lc) * x[159] + wcom * x[160] + (1 / Lc) * (vod32_star - vbd32)
    xdot160 = (-rLc / Lc) * x[160] - wcom * x[159] + (1 / Lc) * (voq32_star - vbq32)

    # ----------------DG33-------------------
    vod33_star = x[397] - nq33 * x[163]
    voq33_star = 0
    xdot161 = x[357] - mp33 * x[162] - wcom
    xdot162 = wc * (vod33_star * x[164] + voq33_star * x[165] - x[162])
    xdot163 = wc * (-vod33_star * x[165] + voq33_star * x[164] - x[163])
    xdot164 = (-rLc / Lc) * x[164] + wcom * x[165] + (1 / Lc) * (vod33_star - vbd33)
    xdot165 = (-rLc / Lc) * x[165] - wcom * x[164] + (1 / Lc) * (voq33_star - vbq33)

    # ----------------DG34-------------------
    vod34_star = x[398] - nq34 * x[168]
    voq34_star = 0
    xdot166 = x[358] - mp34 * x[167] - wcom
    xdot167 = wc * (vod34_star * x[169] + voq34_star * x[170] - x[167])
    xdot168 = wc * (-vod34_star * x[170] + voq34_star * x[169] - x[168])
    xdot169 = (-rLc / Lc) * x[169] + wcom * x[170] + (1 / Lc) * (vod34_star - vbd34)
    xdot170 = (-rLc / Lc) * x[170] - wcom * x[169] + (1 / Lc) * (voq34_star - vbq34)

    # ----------------DG35-------------------
    vod35_star = x[399] - nq35 * x[173]
    voq35_star = 0
    xdot171 = x[359] - mp35 * x[172] - wcom
    xdot172 = wc * (vod35_star * x[174] + voq35_star * x[175] - x[172])
    xdot173 = wc * (-vod35_star * x[175] + voq35_star * x[174] - x[173])
    xdot174 = (-rLc / Lc) * x[174] + wcom * x[175] + (1 / Lc) * (vod35_star - vbd35)
    xdot175 = (-rLc / Lc) * x[175] - wcom * x[174] + (1 / Lc) * (voq35_star - vbq35)

    # ----------------DG36-------------------
    vod36_star = x[400] - nq36 * x[178]
    voq36_star = 0
    xdot176 = x[360] - mp36 * x[177] - wcom
    xdot177 = wc * (vod36_star * x[179] + voq36_star * x[180] - x[177])
    xdot178 = wc * (-vod36_star * x[180] + voq36_star * x[179] - x[178])
    xdot179 = (-rLc / Lc) * x[179] + wcom * x[180] + (1 / Lc) * (vod36_star - vbd36)
    xdot180 = (-rLc / Lc) * x[180] - wcom * x[179] + (1 / Lc) * (voq36_star - vbq36)

    # ----------------DG37-------------------
    vod37_star = x[401] - nq37 * x[183]
    voq37_star = 0
    xdot181 = x[361] - mp37 * x[182] - wcom
    xdot182 = wc * (vod37_star * x[184] + voq37_star * x[185] - x[182])
    xdot183 = wc * (-vod37_star * x[185] + voq37_star * x[184] - x[183])
    xdot184 = (-rLc / Lc) * x[184] + wcom * x[185] + (1 / Lc) * (vod37_star - vbd37)
    xdot185 = (-rLc / Lc) * x[185] - wcom * x[184] + (1 / Lc) * (voq37_star - vbq37)

    # ----------------DG38-------------------
    vod38_star = x[402] - nq38 * x[188]
    voq38_star = 0
    xdot186 = x[362] - mp38 * x[187] - wcom
    xdot187 = wc * (vod38_star * x[189] + voq38_star * x[190] - x[187])
    xdot188 = wc * (-vod38_star * x[190] + voq38_star * x[189] - x[188])
    xdot189 = (-rLc / Lc) * x[189] + wcom * x[190] + (1 / Lc) * (vod38_star - vbd38)
    xdot190 = (-rLc / Lc) * x[190] - wcom * x[189] + (1 / Lc) * (voq38_star - vbq38)

    # ----------------DG39-------------------
    vod39_star = x[403] - nq39 * x[193]
    voq39_star = 0
    xdot191 = x[363] - mp39 * x[192] - wcom
    xdot192 = wc * (vod39_star * x[194] + voq39_star * x[195] - x[192])
    xdot193 = wc * (-vod39_star * x[195] + voq39_star * x[194] - x[193])
    xdot194 = (-rLc / Lc) * x[194] + wcom * x[195] + (1 / Lc) * (vod39_star - vbd39)
    xdot195 = (-rLc / Lc) * x[195] - wcom * x[194] + (1 / Lc) * (voq39_star - vbq39)

    # ----------------DG40-------------------
    vod40_star = x[404] - nq40 * x[198]
    voq40_star = 0
    xdot196 = x[364] - mp40 * x[197] - wcom
    xdot197 = wc * (vod40_star * x[199] + voq40_star * x[200] - x[197])
    xdot198 = wc * (-vod40_star * x[200] + voq40_star * x[199] - x[198])
    xdot199 = (-rLc / Lc) * x[199] + wcom * x[200] + (1 / Lc) * (vod40_star - vbd40)
    xdot200 = (-rLc / Lc) * x[200] - wcom * x[199] + (1 / Lc) * (voq40_star - vbq40)

    # -------------------------Lines------------------
    # ----1 -> 6-----
    xdot201 = (-rline1 / Lline1) * x[201] + wcom * x[202] + (1 / Lline1) * (vbD1 - vbD6)
    xdot202 = (-rline1 / Lline1) * x[202] - wcom * x[201] + (1 / Lline1) * (vbQ1 - vbQ6)
    # ----6 -> 7-----
    xdot203 = (-rline2 / Lline2) * x[203] + wcom * x[204] + (1 / Lline2) * (vbD6 - vbD7)
    xdot204 = (-rline2 / Lline2) * x[204] - wcom * x[203] + (1 / Lline2) * (vbQ6 - vbQ7)
    # ----7 -> 2-----
    xdot205 = (-rline3 / Lline3) * x[205] + wcom * x[206] + (1 / Lline3) * (vbD7 - vbD2)
    xdot206 = (-rline3 / Lline3) * x[206] - wcom * x[205] + (1 / Lline3) * (vbQ7 - vbQ2)
    # ----2 -> 3-----
    xdot207 = (-rline4 / Lline4) * x[207] + wcom * x[208] + (1 / Lline4) * (vbD2 - vbD3)
    xdot208 = (-rline4 / Lline4) * x[208] - wcom * x[207] + (1 / Lline4) * (vbQ2 - vbQ3)
    # ----3 -> 8-----
    xdot209 = (-rline5 / Lline5) * x[209] + wcom * x[210] + (1 / Lline5) * (vbD3 - vbD8)
    xdot210 = (-rline5 / Lline5) * x[210] - wcom * x[209] + (1 / Lline5) * (vbQ3 - vbQ8)
    # ----8 -> 9-----
    xdot211 = (-rline6 / Lline6) * x[211] + wcom * x[212] + (1 / Lline6) * (vbD8 - vbD9)
    xdot212 = (-rline6 / Lline6) * x[212] - wcom * x[211] + (1 / Lline6) * (vbQ8 - vbQ9)
    # ----9 -> 4-----
    xdot213 = (-rline7 / Lline7) * x[213] + wcom * x[214] + (1 / Lline7) * (vbD9 - vbD4)
    xdot214 = (-rline7 / Lline7) * x[214] - wcom * x[213] + (1 / Lline7) * (vbQ9 - vbQ4)
    # ----4 -> 5-----
    xdot215 = (-rline8 / Lline8) * x[215] + wcom * x[216] + (1 / Lline8) * (vbD4 - vbD5)
    xdot216 = (-rline8 / Lline8) * x[216] - wcom * x[215] + (1 / Lline8) * (vbQ4 - vbQ5)
    # ----5 -> 10-----
    xdot217 = (-rline9 / Lline9) * x[217] + wcom * x[218] + (1 / Lline9) * (vbD5 - vbD10)
    xdot218 = (-rline9 / Lline9) * x[218] - wcom * x[217] + (1 / Lline9) * (vbQ5 - vbQ10)
    # ----8 -> 11-----
    xdot219 = (-rline10 / Lline10) * x[219] + wcom * x[220] + (1 / Lline10) * (vbD8 - vbD11)
    xdot220 = (-rline10 / Lline10) * x[220] - wcom * x[219] + (1 / Lline10) * (vbQ8 - vbQ11)
    # ----10 -> 13-----
    xdot221 = (-rline11 / Lline11) * x[221] + wcom * x[222] + (1 / Lline11) * (vbD10 - vbD13)
    xdot222 = (-rline11 / Lline11) * x[222] - wcom * x[221] + (1 / Lline11) * (vbQ10 - vbQ13)
    # ----11 -> 16-----
    xdot223 = (-rline12 / Lline12) * x[223] + wcom * x[224] + (1 / Lline12) * (vbD11 - vbD16)
    xdot224 = (-rline12 / Lline12) * x[224] - wcom * x[223] + (1 / Lline12) * (vbQ11 - vbQ16)
    # ----16 -> 17-----
    xdot225 = (-rline13 / Lline13) * x[225] + wcom * x[226] + (1 / Lline13) * (vbD16 - vbD17)
    xdot226 = (-rline13 / Lline13) * x[226] - wcom * x[225] + (1 / Lline13) * (vbQ16 - vbQ17)
    # ----17 -> 12-----
    xdot227 = (-rline14 / Lline14) * x[227] + wcom * x[228] + (1 / Lline14) * (vbD17 - vbD12)
    xdot228 = (-rline14 / Lline14) * x[228] - wcom * x[227] + (1 / Lline14) * (vbQ17 - vbQ12)
    # ----12 -> 13-----
    xdot229 = (-rline15 / Lline15) * x[229] + wcom * x[230] + (1 / Lline15) * (vbD12 - vbD13)
    xdot230 = (-rline15 / Lline15) * x[230] - wcom * x[229] + (1 / Lline15) * (vbQ12 - vbQ13)
    # ----13 -> 18-----
    xdot231 = (-rline16 / Lline16) * x[231] + wcom * x[232] + (1 / Lline16) * (vbD13 - vbD18)
    xdot232 = (-rline16 / Lline16) * x[232] - wcom * x[231] + (1 / Lline16) * (vbQ13 - vbQ18)
    # ----18 -> 19-----
    xdot233 = (-rline17 / Lline17) * x[233] + wcom * x[234] + (1 / Lline17) * (vbD18 - vbD19)
    xdot234 = (-rline17 / Lline17) * x[234] - wcom * x[233] + (1 / Lline17) * (vbQ18 - vbQ19)
    # ----19 -> 14-----
    xdot235 = (-rline18 / Lline18) * x[235] + wcom * x[236] + (1 / Lline18) * (vbD19 - vbD14)
    xdot236 = (-rline18 / Lline18) * x[236] - wcom * x[235] + (1 / Lline18) * (vbQ19 - vbQ14)
    # ----14 -> 15-----
    xdot237 = (-rline19 / Lline19) * x[237] + wcom * x[238] + (1 / Lline19) * (vbD14 - vbD15)
    xdot238 = (-rline19 / Lline19) * x[238] - wcom * x[237] + (1 / Lline19) * (vbQ14 - vbQ15)
    # ----15 -> 20-----
    xdot239 = (-rline20 / Lline20) * x[239] + wcom * x[240] + (1 / Lline20) * (vbD15 - vbD20)
    xdot240 = (-rline20 / Lline20) * x[240] - wcom * x[239] + (1 / Lline20) * (vbQ15 - vbQ20)
    # ----21 -> 26-----
    xdot241 = (-rline21 / Lline21) * x[241] + wcom * x[242] + (1 / Lline21) * (vbD21 - vbD26)
    xdot242 = (-rline21 / Lline21) * x[242] - wcom * x[241] + (1 / Lline21) * (vbQ21 - vbQ26)
    # ----26 -> 27-----
    xdot243 = (-rline22 / Lline22) * x[243] + wcom * x[244] + (1 / Lline22) * (vbD26 - vbD27)
    xdot244 = (-rline22 / Lline22) * x[244] - wcom * x[243] + (1 / Lline22) * (vbQ26 - vbQ27)
    # ----27 -> 22-----
    xdot245 = (-rline23 / Lline23) * x[245] + wcom * x[246] + (1 / Lline23) * (vbD27 - vbD22)
    xdot246 = (-rline23 / Lline23) * x[246] - wcom * x[245] + (1 / Lline23) * (vbQ27 - vbQ22)
    # ----22 -> 23-----
    xdot247 = (-rline24 / Lline24) * x[247] + wcom * x[248] + (1 / Lline24) * (vbD22 - vbD23)
    xdot248 = (-rline24 / Lline24) * x[248] - wcom * x[247] + (1 / Lline24) * (vbQ22 - vbQ23)
    # ----23 -> 28-----
    xdot249 = (-rline25 / Lline25) * x[249] + wcom * x[250] + (1 / Lline25) * (vbD23 - vbD28)
    xdot250 = (-rline25 / Lline25) * x[250] - wcom * x[249] + (1 / Lline25) * (vbQ23 - vbQ28)
    # ----28 -> 29-----
    xdot251 = (-rline26 / Lline26) * x[251] + wcom * x[252] + (1 / Lline26) * (vbD28 - vbD29)
    xdot252 = (-rline26 / Lline26) * x[252] - wcom * x[251] + (1 / Lline26) * (vbQ28 - vbQ29)
    # ----29 -> 24-----
    xdot253 = (-rline27 / Lline27) * x[253] + wcom * x[254] + (1 / Lline27) * (vbD29 - vbD24)
    xdot254 = (-rline27 / Lline27) * x[254] - wcom * x[253] + (1 / Lline27) * (vbQ29 - vbQ24)
    # ----24 -> 25-----
    xdot255 = (-rline28 / Lline28) * x[255] + wcom * x[256] + (1 / Lline28) * (vbD24 - vbD25)
    xdot256 = (-rline28 / Lline28) * x[256] - wcom * x[255] + (1 / Lline28) * (vbQ24 - vbQ25)
    # ----25 -> 30-----
    xdot257 = (-rline29 / Lline29) * x[257] + wcom * x[258] + (1 / Lline29) * (vbD25 - vbD30)
    xdot258 = (-rline29 / Lline29) * x[258] - wcom * x[257] + (1 / Lline29) * (vbQ25 - vbQ30)
    # ----28 -> 31-----
    xdot259 = (-rline30 / Lline30) * x[259] + wcom * x[260] + (1 / Lline30) * (vbD28 - vbD31)
    xdot260 = (-rline30 / Lline30) * x[260] - wcom * x[259] + (1 / Lline30) * (vbQ28 - vbQ31)
    # ----30 -> 33-----
    xdot261 = (-rline31 / Lline31) * x[261] + wcom * x[262] + (1 / Lline31) * (vbD30 - vbD33)
    xdot262 = (-rline31 / Lline31) * x[262] - wcom * x[261] + (1 / Lline31) * (vbQ30 - vbQ33)
    # ----31 -> 36-----
    xdot263 = (-rline32 / Lline32) * x[263] + wcom * x[264] + (1 / Lline32) * (vbD31 - vbD36)
    xdot264 = (-rline32 / Lline32) * x[264] - wcom * x[263] + (1 / Lline32) * (vbQ31 - vbQ36)
    # ----36 -> 37-----
    xdot265 = (-rline33 / Lline33) * x[265] + wcom * x[266] + (1 / Lline33) * (vbD36 - vbD37)
    xdot266 = (-rline33 / Lline33) * x[266] - wcom * x[265] + (1 / Lline33) * (vbQ36 - vbQ37)
    # ----37 -> 32-----
    xdot267 = (-rline34 / Lline34) * x[267] + wcom * x[268] + (1 / Lline34) * (vbD37 - vbD32)
    xdot268 = (-rline34 / Lline34) * x[268] - wcom * x[267] + (1 / Lline34) * (vbQ37 - vbQ32)
    # ----32 -> 33-----
    xdot269 = (-rline35 / Lline35) * x[269] + wcom * x[270] + (1 / Lline35) * (vbD32 - vbD33)
    xdot270 = (-rline35 / Lline35) * x[270] - wcom * x[269] + (1 / Lline35) * (vbQ32 - vbQ33)
    # ----33 -> 38-----
    xdot271 = (-rline36 / Lline36) * x[271] + wcom * x[272] + (1 / Lline36) * (vbD33 - vbD38)
    xdot272 = (-rline36 / Lline36) * x[272] - wcom * x[271] + (1 / Lline36) * (vbQ33 - vbQ38)
    # ----38 -> 39-----
    xdot273 = (-rline37 / Lline37) * x[273] + wcom * x[274] + (1 / Lline37) * (vbD38 - vbD39)
    xdot274 = (-rline37 / Lline37) * x[274] - wcom * x[273] + (1 / Lline37) * (vbQ38 - vbQ39)
    # ----39 -> 34-----
    xdot275 = (-rline38 / Lline38) * x[275] + wcom * x[276] + (1 / Lline38) * (vbD39 - vbD34)
    xdot276 = (-rline38 / Lline38) * x[276] - wcom * x[275] + (1 / Lline38) * (vbQ39 - vbQ34)
    # ----34 -> 35-----
    xdot277 = (-rline39 / Lline39) * x[277] + wcom * x[278] + (1 / Lline39) * (vbD34 - vbD35)
    xdot278 = (-rline39 / Lline39) * x[278] - wcom * x[277] + (1 / Lline39) * (vbQ34 - vbQ35)
    # ----35 -> 30-----
    xdot279 = (-rline40 / Lline40) * x[279] + wcom * x[280] + (1 / Lline40) * (vbD35 - vbD40)
    xdot280 = (-rline40 / Lline40) * x[280] - wcom * x[279] + (1 / Lline40) * (vbQ35 - vbQ40)
    # ----18 -> 21-----
    xdot281 = (-rline41 / Lline41) * x[281] + wcom * x[282] + (1 / Lline41) * (vbD18 - vbD21)
    xdot282 = (-rline41 / Lline41) * x[282] - wcom * x[281] + (1 / Lline41) * (vbQ18 - vbQ21)
    # ----20 -> 23-----
    xdot283 = (-rline42 / Lline42) * x[283] + wcom * x[284] + (1 / Lline42) * (vbD20 - vbD23)
    xdot284 = (-rline42 / Lline42) * x[284] - wcom * x[283] + (1 / Lline42) * (vbQ20 - vbQ23)

    # -------------------------Loads------------------
    # ------Load1--------
    xdot285 = (-disturbance_R[0] * Rload1 / (disturbance_L[0] * Lload1)) * x[285] + wcom * x[286] + (
            1 / (disturbance_L[0] * Lload1)) * vbD1
    xdot286 = (-disturbance_R[0] * Rload1 / (disturbance_L[0] * Lload1)) * x[286] - wcom * x[285] + (
            1 / (disturbance_L[0] * Lload1)) * vbQ1
    # ------Load2--------
    xdot287 = (-disturbance_R[1] * Rload2 / (disturbance_L[1] * Lload2)) * x[287] + wcom * x[288] + (
            1 / (disturbance_L[1] * Lload2)) * vbD7
    xdot288 = (-disturbance_R[1] * Rload2 / (disturbance_L[1] * Lload2)) * x[288] - wcom * x[287] + (
            1 / (disturbance_L[1] * Lload2)) * vbQ7
    # ------Load3--------
    xdot289 = (-disturbance_R[2] * Rload3 / (disturbance_L[2] * Lload3)) * x[289] + wcom * x[290] + (
            1 / (disturbance_L[2] * Lload3)) * vbD3
    xdot290 = (-disturbance_R[2] * Rload3 / (disturbance_L[2] * Lload3)) * x[290] - wcom * x[289] + (
            1 / (disturbance_L[2] * Lload3)) * vbQ3
    # ------Load4--------
    xdot291 = (-disturbance_R[3] * Rload4 / (disturbance_L[3] * Lload4)) * x[291] + wcom * x[292] + (
            1 / (disturbance_L[3] * Lload4)) * vbD9
    xdot292 = (-disturbance_R[3] * Rload4 / (disturbance_L[3] * Lload4)) * x[292] - wcom * x[291] + (
            1 / (disturbance_L[3] * Lload4)) * vbQ9
    # ------Load5--------
    xdot293 = (-disturbance_R[4] * Rload5 / (disturbance_L[4] * Lload5)) * x[293] + wcom * x[294] + (
            1 / (disturbance_L[4] * Lload5)) * vbD5
    xdot294 = (-disturbance_R[4] * Rload5 / (disturbance_L[4] * Lload5)) * x[294] - wcom * x[293] + (
            1 / (disturbance_L[4] * Lload5)) * vbQ5
    # ------Load6--------
    xdot295 = (-disturbance_R[5] * Rload6 / (disturbance_L[5] * Lload6)) * x[295] + wcom * x[296] + (
            1 / (disturbance_L[5] * Lload6)) * vbD11
    xdot296 = (-disturbance_R[5] * Rload6 / (disturbance_L[5] * Lload6)) * x[296] - wcom * x[295] + (
            1 / (disturbance_L[5] * Lload6)) * vbQ11
    # ------Load7--------
    xdot297 = (-disturbance_R[6] * Rload7 / (disturbance_L[6] * Lload7)) * x[297] + wcom * x[298] + (
            1 / (disturbance_L[6] * Lload7)) * vbD17
    xdot298 = (-disturbance_R[6] * Rload7 / (disturbance_L[6] * Lload7)) * x[298] - wcom * x[297] + (
            1 / (disturbance_L[6] * Lload7)) * vbQ17
    # ------Load8--------
    xdot299 = (-disturbance_R[7] * Rload8 / (disturbance_L[7] * Lload8)) * x[299] + wcom * x[300] + (
            1 / (disturbance_L[7] * Lload8)) * vbD13
    xdot300 = (-disturbance_R[7] * Rload8 / (disturbance_L[7] * Lload8)) * x[300] - wcom * x[299] + (
            1 / (disturbance_L[7] * Lload8)) * vbQ13
    # ------Load9--------
    xdot301 = (-disturbance_R[8] * Rload9 / (disturbance_L[8] * Lload9)) * x[301] + wcom * x[302] + (
            1 / (disturbance_L[8] * Lload9)) * vbD19
    xdot302 = (-disturbance_R[8] * Rload9 / (disturbance_L[8] * Lload9)) * x[302] - wcom * x[301] + (
            1 / (disturbance_L[8] * Lload9)) * vbQ19
    # ------Load10--------
    xdot303 = (-disturbance_R[9] * Rload10 / (disturbance_L[9] * Lload10)) * x[303] + wcom * x[304] + (
            1 / (disturbance_L[9] * Lload10)) * vbD15
    xdot304 = (-disturbance_R[9] * Rload10 / (disturbance_L[9] * Lload10)) * x[304] - wcom * x[303] + (
            1 / (disturbance_L[9] * Lload10)) * vbQ15

    # ------Load11--------
    xdot305 = (-disturbance_R[10] * Rload11 / (disturbance_L[10] * Lload11)) * x[305] + wcom * x[306] + (
            1 / (disturbance_L[10] * Lload11)) * vbD21
    xdot306 = (-disturbance_R[10] * Rload11 / (disturbance_L[10] * Lload1)) * x[306] - wcom * x[305] + (
            1 / (disturbance_L[10] * Lload11)) * vbQ21
    # ------Load12--------
    xdot307 = (-disturbance_R[11] * Rload12 / (disturbance_L[11] * Lload12)) * x[307] + wcom * x[308] + (
            1 / (disturbance_L[11] * Lload12)) * vbD27
    xdot308 = (-disturbance_R[11] * Rload12 / (disturbance_L[11] * Lload12)) * x[308] - wcom * x[307] + (
            1 / (disturbance_L[11] * Lload12)) * vbQ27
    # ------Load13--------
    xdot309 = (-disturbance_R[12] * Rload13 / (disturbance_L[12] * Lload13)) * x[309] + wcom * x[310] + (
            1 / (disturbance_L[12] * Lload13)) * vbD23
    xdot310 = (-disturbance_R[12] * Rload13 / (disturbance_L[12] * Lload13)) * x[310] - wcom * x[309] + (
            1 / (disturbance_L[12] * Lload13)) * vbQ23
    # ------Load14--------
    xdot311 = (-disturbance_R[13] * Rload14 / (disturbance_L[13] * Lload14)) * x[311] + wcom * x[312] + (
            1 / (disturbance_L[13] * Lload14)) * vbD29
    xdot312 = (-disturbance_R[13] * Rload14 / (disturbance_L[13] * Lload14)) * x[312] - wcom * x[311] + (
            1 / (disturbance_L[13] * Lload14)) * vbQ29
    # ------Load15--------
    xdot313 = (-disturbance_R[14] * Rload15 / (disturbance_L[14] * Lload15)) * x[313] + wcom * x[314] + (
            1 / (disturbance_L[14] * Lload15)) * vbD25
    xdot314 = (-disturbance_R[14] * Rload15 / (disturbance_L[14] * Lload15)) * x[314] - wcom * x[313] + (
            1 / (disturbance_L[14] * Lload15)) * vbQ25
    # ------Load16--------
    xdot315 = (-disturbance_R[15] * Rload16 / (disturbance_L[15] * Lload16)) * x[315] + wcom * x[316] + (
            1 / (disturbance_L[15] * Lload16)) * vbD31
    xdot316 = (-disturbance_R[15] * Rload16 / (disturbance_L[15] * Lload16)) * x[316] - wcom * x[315] + (
            1 / (disturbance_L[15] * Lload16)) * vbQ31
    # ------Load17--------
    xdot317 = (-disturbance_R[16] * Rload17 / (disturbance_L[16] * Lload17)) * x[317] + wcom * x[318] + (
            1 / (disturbance_L[16] * Lload17)) * vbD37
    xdot318 = (-disturbance_R[16] * Rload17 / (disturbance_L[16] * Lload17)) * x[318] - wcom * x[317] + (
            1 / (disturbance_L[16] * Lload17)) * vbQ37
    # ------Load18--------
    xdot319 = (-disturbance_R[17] * Rload18 / (disturbance_L[17] * Lload18)) * x[319] + wcom * x[320] + (
            1 / (disturbance_L[17] * Lload18)) * vbD33
    xdot320 = (-disturbance_R[17] * Rload18 / (disturbance_L[17] * Lload18)) * x[320] - wcom * x[319] + (
            1 / (disturbance_L[17] * Lload18)) * vbQ33
    # ------Load19--------
    xdot321 = (-disturbance_R[18] * Rload19 / (disturbance_L[18] * Lload19)) * x[321] + wcom * x[322] + (
            1 / (disturbance_L[18] * Lload19)) * vbD39
    xdot322 = (-disturbance_R[18] * Rload19 / (disturbance_L[18] * Lload19)) * x[322] - wcom * x[321] + (
            1 / (disturbance_L[18] * Lload19)) * vbQ39
    # ------Load20--------
    xdot323 = (-disturbance_R[19] * Rload20 / (disturbance_L[19] * Lload20)) * x[323] + wcom * x[324] + (
            1 / (disturbance_L[19] * Lload20)) * vbD35
    xdot324 = (-disturbance_R[19] * Rload20 / (disturbance_L[19] * Lload20)) * x[324] - wcom * x[323] + (
            1 / (disturbance_L[19] * Lload20)) * vbQ35

    # ----------------------------------------------------
    # Controller Parameters
    if t <= 0.4:
        xdot325 = 0
        xdot326 = 0
        xdot327 = 0
        xdot328 = 0
        xdot329 = 0
        xdot330 = 0
        xdot331 = 0
        xdot332 = 0
        xdot333 = 0
        xdot334 = 0
        xdot335 = 0
        xdot336 = 0
        xdot337 = 0
        xdot338 = 0
        xdot339 = 0
        xdot340 = 0
        xdot341 = 0
        xdot342 = 0
        xdot343 = 0
        xdot344 = 0
        xdot345 = 0
        xdot346 = 0
        xdot347 = 0
        xdot348 = 0
        xdot349 = 0
        xdot350 = 0
        xdot351 = 0
        xdot352 = 0
        xdot353 = 0
        xdot354 = 0
        xdot355 = 0
        xdot356 = 0
        xdot357 = 0
        xdot358 = 0
        xdot359 = 0
        xdot360 = 0
        xdot361 = 0
        xdot362 = 0
        xdot363 = 0
        xdot364 = 0

        xdot365 = 0
        xdot366 = 0
        xdot367 = 0
        xdot368 = 0
        xdot369 = 0
        xdot370 = 0
        xdot371 = 0
        xdot372 = 0
        xdot373 = 0
        xdot374 = 0
        xdot375 = 0
        xdot376 = 0
        xdot377 = 0
        xdot378 = 0
        xdot379 = 0
        xdot380 = 0
        xdot381 = 0
        xdot382 = 0
        xdot383 = 0
        xdot384 = 0
        xdot385 = 0
        xdot386 = 0
        xdot387 = 0
        xdot388 = 0
        xdot389 = 0
        xdot390 = 0
        xdot391 = 0
        xdot392 = 0
        xdot393 = 0
        xdot394 = 0
        xdot395 = 0
        xdot396 = 0
        xdot397 = 0
        xdot398 = 0
        xdot399 = 0
        xdot400 = 0
        xdot401 = 0
        xdot402 = 0
        xdot403 = 0
        xdot404 = 0
    else:
        Pratio = np.array([[mp1 * x[2]], [mp2 * x[7]], [mp3 * x[12]], [mp4 * x[17]], [mp5 * x[22]],
                           [mp6 * x[27]], [mp7 * x[32]], [mp8 * x[37]], [mp9 * x[42]], [mp10 * x[47]],
                           [mp11 * x[52]], [mp12 * x[57]], [mp13 * x[62]], [mp14 * x[67]], [mp15 * x[72]],
                           [mp16 * x[77]], [mp17 * x[82]], [mp18 * x[87]], [mp19 * x[92]], [mp20 * x[97]],
                           [mp21 * x[102]], [mp22 * x[107]], [mp23 * x[112]], [mp24 * x[117]], [mp25 * x[122]],
                           [mp26 * x[127]], [mp27 * x[132]], [mp28 * x[137]], [mp29 * x[142]], [mp30 * x[147]],
                           [mp31 * x[152]], [mp32 * x[157]], [mp33 * x[162]], [mp34 * x[167]], [mp35 * x[172]],
                           [mp36 * x[177]], [mp37 * x[182]], [mp38 * x[187]], [mp39 * x[192]], [mp40 * x[197]]])

        w_array = np.array(
            [[x[325] - Pratio[0][0]], [x[326] - Pratio[1][0]], [x[327] - Pratio[2][0]], [x[328] - Pratio[3][0]],
             [x[329] - Pratio[4][0]], [x[330] - Pratio[5][0]], [x[331] - Pratio[6][0]], [x[332] - Pratio[7][0]],
             [x[333] - Pratio[8][0]], [x[334] - Pratio[9][0]], [x[335] - Pratio[10][0]], [x[336] - Pratio[11][0]],
             [x[337] - Pratio[12][0]], [x[338] - Pratio[13][0]], [x[339] - Pratio[14][0]], [x[340] - Pratio[15][0]],
             [x[341] - Pratio[16][0]], [x[342] - Pratio[17][0]], [x[343] - Pratio[18][0]], [x[344] - Pratio[19][0]],
             [x[345] - Pratio[20][0]], [x[346] - Pratio[21][0]], [x[347] - Pratio[22][0]], [x[348] - Pratio[23][0]],
             [x[349] - Pratio[24][0]], [x[350] - Pratio[25][0]], [x[351] - Pratio[26][0]], [x[352] - Pratio[27][0]],
             [x[353] - Pratio[28][0]], [x[354] - Pratio[29][0]], [x[355] - Pratio[30][0]], [x[356] - Pratio[31][0]],
             [x[357] - Pratio[32][0]], [x[358] - Pratio[33][0]], [x[359] - Pratio[34][0]], [x[360] - Pratio[35][0]],
             [x[361] - Pratio[36][0]], [x[362] - Pratio[37][0]], [x[363] - Pratio[38][0]], [x[364] - Pratio[39][0]]
             ])

        # Conventional Freq Control
        Synch_Mat = -1 * a_ctrl * (
                np.dot(L + G, w_array - np.array([[wref], [wref], [wref], [wref], [wref], [wref], [wref], [wref],
                                                  [wref], [wref], [wref], [wref], [wref], [wref], [wref], [wref],
                                                  [wref], [wref], [wref], [wref], [wref], [wref], [wref], [wref],
                                                  [wref], [wref], [wref], [wref], [wref], [wref], [wref], [wref],
                                                  [wref], [wref], [wref], [wref], [wref], [wref], [wref], [wref]]))
                + np.dot(L, Pratio))

        # ---frequency input---
        xdot325 = Synch_Mat[0][0]
        xdot326 = Synch_Mat[1][0]
        xdot327 = Synch_Mat[2][0]
        xdot328 = Synch_Mat[3][0]
        xdot329 = Synch_Mat[4][0]
        xdot330 = Synch_Mat[5][0]
        xdot331 = Synch_Mat[6][0]
        xdot332 = Synch_Mat[7][0]
        xdot333 = Synch_Mat[8][0]
        xdot334 = Synch_Mat[9][0]
        xdot335 = Synch_Mat[10][0]
        xdot336 = Synch_Mat[11][0]
        xdot337 = Synch_Mat[12][0]
        xdot338 = Synch_Mat[13][0]
        xdot339 = Synch_Mat[14][0]
        xdot340 = Synch_Mat[15][0]
        xdot341 = Synch_Mat[16][0]
        xdot342 = Synch_Mat[17][0]
        xdot343 = Synch_Mat[18][0]
        xdot344 = Synch_Mat[19][0]
        xdot345 = Synch_Mat[20][0]
        xdot346 = Synch_Mat[21][0]
        xdot347 = Synch_Mat[22][0]
        xdot348 = Synch_Mat[23][0]
        xdot349 = Synch_Mat[24][0]
        xdot350 = Synch_Mat[25][0]
        xdot351 = Synch_Mat[26][0]
        xdot352 = Synch_Mat[27][0]
        xdot353 = Synch_Mat[28][0]
        xdot354 = Synch_Mat[29][0]
        xdot355 = Synch_Mat[30][0]
        xdot356 = Synch_Mat[31][0]
        xdot357 = Synch_Mat[32][0]
        xdot358 = Synch_Mat[33][0]
        xdot359 = Synch_Mat[34][0]
        xdot360 = Synch_Mat[35][0]
        xdot361 = Synch_Mat[36][0]
        xdot362 = Synch_Mat[37][0]
        xdot363 = Synch_Mat[38][0]
        xdot364 = Synch_Mat[39][0]

        # ---voltage input---
        xdot365 = 0
        xdot366 = 0
        xdot367 = 0
        xdot368 = 0
        xdot369 = 0
        xdot370 = 0
        xdot371 = 0
        xdot372 = 0
        xdot373 = 0
        xdot374 = 0
        xdot375 = 0
        xdot376 = 0
        xdot377 = 0
        xdot378 = 0
        xdot379 = 0
        xdot380 = 0
        xdot381 = 0
        xdot382 = 0
        xdot383 = 0
        xdot384 = 0
        xdot385 = 0
        xdot386 = 0
        xdot387 = 0
        xdot388 = 0
        xdot389 = 0
        xdot390 = 0
        xdot391 = 0
        xdot392 = 0
        xdot393 = 0
        xdot394 = 0
        xdot395 = 0
        xdot396 = 0
        xdot397 = 0
        xdot398 = 0
        xdot399 = 0
        xdot400 = 0
        xdot401 = 0
        xdot402 = 0
        xdot403 = 0
        xdot404 = 0

    return np.array(
        [0, xdot1, xdot2, xdot3, xdot4, xdot5, xdot6, xdot7, xdot8, xdot9, xdot10, xdot11, xdot12, xdot13, xdot14,
         xdot15, xdot16, xdot17, xdot18, xdot19, xdot20, xdot21, xdot22, xdot23, xdot24, xdot25, xdot26, xdot27,
         xdot28, xdot29, xdot30, xdot31, xdot32, xdot33, xdot34, xdot35, xdot36, xdot37, xdot38, xdot39, xdot40,
         xdot41, xdot42, xdot43, xdot44, xdot45, xdot46, xdot47, xdot48, xdot49, xdot50, xdot51, xdot52, xdot53,
         xdot54, xdot55, xdot56, xdot57, xdot58, xdot59, xdot60, xdot61, xdot62, xdot63, xdot64, xdot65, xdot66,
         xdot67, xdot68, xdot69, xdot70, xdot71, xdot72, xdot73, xdot74, xdot75, xdot76, xdot77, xdot78, xdot79,
         xdot80, xdot81, xdot82, xdot83, xdot84, xdot85, xdot86, xdot87, xdot88, xdot89, xdot90, xdot91, xdot92,
         xdot93, xdot94, xdot95, xdot96, xdot97, xdot98, xdot99, xdot100, xdot101, xdot102, xdot103, xdot104, xdot105,
         xdot106, xdot107, xdot108, xdot109, xdot110, xdot111, xdot112, xdot113, xdot114,
         xdot115, xdot116, xdot117, xdot118, xdot119, xdot120, xdot121, xdot122, xdot123, xdot124, xdot125, xdot126,
         xdot127, xdot128, xdot129, xdot130, xdot131, xdot132, xdot133, xdot134, xdot135, xdot136, xdot137, xdot138,
         xdot139, xdot140, xdot141, xdot142, xdot143, xdot144, xdot145, xdot146, xdot147, xdot148, xdot149, xdot150,
         xdot151, xdot152, xdot153, xdot154, xdot155, xdot156, xdot157, xdot158, xdot159, xdot160, xdot161, xdot162,
         xdot163, xdot164, xdot165, xdot166, xdot167, xdot168, xdot169, xdot170, xdot171, xdot172, xdot173, xdot174,
         xdot175, xdot176, xdot177, xdot178, xdot179, xdot180, xdot181, xdot182, xdot183, xdot184, xdot185, xdot186,
         xdot187, xdot188, xdot189, xdot190, xdot191, xdot192, xdot193, xdot194, xdot195, xdot196, xdot197,
         xdot198, xdot199, xdot200, xdot201, xdot202, xdot203, xdot204,
         xdot205, xdot206, xdot207, xdot208, xdot209, xdot210, xdot211, xdot212, xdot213, xdot214, xdot215,
         xdot216, xdot217, xdot218, xdot219, xdot220, xdot221, xdot222, xdot223, xdot224, xdot225, xdot226,
         xdot227, xdot228, xdot229, xdot230, xdot231, xdot232, xdot233, xdot234, xdot235, xdot236, xdot237,
         xdot238, xdot239, xdot240, xdot241, xdot242, xdot243, xdot244, xdot245, xdot246, xdot247, xdot248, xdot249,
         xdot250, xdot251, xdot252, xdot253, xdot254, xdot255, xdot256, xdot257, xdot258, xdot259, xdot260,
         xdot261, xdot262, xdot263, xdot264, xdot265, xdot266, xdot267, xdot268, xdot269, xdot270, xdot271,
         xdot272, xdot273, xdot274, xdot275, xdot276, xdot277, xdot278, xdot279, xdot280, xdot281, xdot282,
         xdot283, xdot284, xdot285, xdot286, xdot287, xdot288, xdot289, xdot290, xdot291, xdot292,
         xdot293, xdot294, xdot295, xdot296, xdot297, xdot298, xdot299, xdot300, xdot301, xdot302, xdot303,
         xdot304, xdot305, xdot306, xdot307, xdot308, xdot309, xdot310, xdot311, xdot312, xdot313, xdot314,
         xdot315, xdot316, xdot317, xdot318, xdot319, xdot320, xdot321, xdot322, xdot323, xdot324,
         xdot325, xdot326, xdot327, xdot328, xdot329, xdot330, xdot331, xdot332, xdot333, xdot334,
         xdot335, xdot336, xdot337, xdot338, xdot339, xdot340, xdot341, xdot342, xdot343, xdot344, xdot345, xdot346,
         xdot347, xdot348, xdot349, xdot350, xdot351, xdot352, xdot353,
         xdot354, xdot355, xdot356, xdot357, xdot358, xdot359, xdot360, xdot361, xdot362, xdot363, xdot364,
         xdot365, xdot366, xdot367, xdot368, xdot369, xdot370, xdot371, xdot372, xdot373, xdot374, xdot375, xdot376,
         xdot377, xdot378, xdot379, xdot380, xdot381, xdot382, xdot383, xdot384, xdot385, xdot386, xdot387, xdot388,
         xdot389, xdot390, xdot391, xdot392, xdot393, xdot394, xdot395, xdot396, xdot397, xdot398, xdot399, xdot400,
         xdot401, xdot402, xdot403, xdot404,
         ioD1, ioQ1, vbD1, vbQ1, ioD2, ioQ2, vbD2, vbQ2, ioD3, ioQ3, vbD3, vbQ3, ioD4, ioQ4, vbD4, vbQ4,
         ioD5, ioQ5, vbD5, vbQ5, ioD6, ioQ6, vbD6, vbQ6, ioD7, ioQ7, vbD7, vbQ7, ioD8, ioQ8, vbD8, vbQ8,
         ioD9, ioQ9, vbD9, vbQ9, ioD10, ioQ10, vbD10, vbQ10, ioD11, ioQ11, vbD11, vbQ11, ioD12, ioQ12, vbD12, vbQ12,
         ioD13, ioQ13, vbD13, vbQ13, ioD14, ioQ14, vbD14, vbQ14, ioD15, ioQ15, vbD15, vbQ15,
         ioD16, ioQ16, vbD16, vbQ16, ioD17, ioQ17, vbD17, vbQ17, ioD18, ioQ18, vbD18, vbQ18,
         ioD19, ioQ19, vbD19, vbQ19, ioD20, ioQ20, vbD20, vbQ20, ioD21, ioQ21, vbD21, vbQ21,
         ioD22, ioQ22, vbD22, vbQ22, ioD23, ioQ23, vbD23, vbQ23, ioD24, ioQ24, vbD24, vbQ24,
         ioD25, ioQ25, vbD25, vbQ25, ioD26, ioQ26, vbD26, vbQ26, ioD27, ioQ27, vbD27, vbQ27,
         ioD28, ioQ28, vbD28, vbQ28, ioD29, ioQ29, vbD29, vbQ29, ioD30, ioQ30, vbD30, vbQ30,
         ioD31, ioQ31, vbD31, vbQ31, ioD32, ioQ32, vbD32, vbQ32, ioD33, ioQ33, vbD33, vbQ33,
         ioD34, ioQ34, vbD34, vbQ34, ioD35, ioQ35, vbD35, vbQ35, ioD36, ioQ36, vbD36, vbQ36,
         ioD37, ioQ37, vbD37, vbQ37, ioD38, ioQ38, vbD38, vbQ38, ioD39, ioQ39, vbD39, vbQ39,
         ioD40, ioQ40, vbD40, vbQ40
         ])

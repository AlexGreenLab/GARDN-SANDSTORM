��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%��L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.9.22unknown8��	
�
spectral_normalization_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namespectral_normalization_6/bias
�
1spectral_normalization_6/bias/Read/ReadVariableOpReadVariableOpspectral_normalization_6/bias*
_output_shapes
:*
dtype0
�
spectral_normalization_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namespectral_normalization_5/bias
�
1spectral_normalization_5/bias/Read/ReadVariableOpReadVariableOpspectral_normalization_5/bias*
_output_shapes
: *
dtype0
�
spectral_normalization_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_namespectral_normalization_4/bias
�
1spectral_normalization_4/bias/Read/ReadVariableOpReadVariableOpspectral_normalization_4/bias*
_output_shapes
:@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	�*
dtype0
�
spectral_normalization_6/sn_uVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_namespectral_normalization_6/sn_u
�
1spectral_normalization_6/sn_u/Read/ReadVariableOpReadVariableOpspectral_normalization_6/sn_u*
_output_shapes

:*
dtype0
�
spectral_normalization_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!spectral_normalization_6/kernel
�
3spectral_normalization_6/kernel/Read/ReadVariableOpReadVariableOpspectral_normalization_6/kernel*&
_output_shapes
: *
dtype0
�
spectral_normalization_5/sn_uVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *.
shared_namespectral_normalization_5/sn_u
�
1spectral_normalization_5/sn_u/Read/ReadVariableOpReadVariableOpspectral_normalization_5/sn_u*
_output_shapes

: *
dtype0
�
spectral_normalization_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@ *0
shared_name!spectral_normalization_5/kernel
�
3spectral_normalization_5/kernel/Read/ReadVariableOpReadVariableOpspectral_normalization_5/kernel*&
_output_shapes
:	@ *
dtype0
�
spectral_normalization_4/sn_uVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*.
shared_namespectral_normalization_4/sn_u
�
1spectral_normalization_4/sn_u/Read/ReadVariableOpReadVariableOpspectral_normalization_4/sn_u*
_output_shapes

:@*
dtype0
�
spectral_normalization_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!spectral_normalization_4/kernel
�
3spectral_normalization_4/kernel/Read/ReadVariableOpReadVariableOpspectral_normalization_4/kernel*&
_output_shapes
:@*
dtype0

NoOpNoOp
�B
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�A
value�AB�A B�A
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	layer
w
w_shape
sn_u
u*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	layer
 w
!w_shape
"sn_u
"u*
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
	)layer
*w
+w_shape
,sn_u
,u*
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses* 
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

9kernel
:bias*
R
0
;1
2
 3
<4
"5
*6
=7
,8
99
:10*
<
0
;1
 2
<3
*4
=5
96
:7*
* 
�
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ctrace_0
Dtrace_1
Etrace_2
Ftrace_3* 
6
Gtrace_0
Htrace_1
Itrace_2
Jtrace_3* 
* 

Kserving_default* 

0
;1
2*

0
;1*
* 
�
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Qtrace_0
Rtrace_1* 

Strace_0
Ttrace_1* 
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses
[
activation

kernel
;bias
 \_jit_compiled_convolution_op*
jd
VARIABLE_VALUEspectral_normalization_4/kernel1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUE*
* 
ke
VARIABLE_VALUEspectral_normalization_4/sn_u4layer_with_weights-0/sn_u/.ATTRIBUTES/VARIABLE_VALUE*

 0
<1
"2*

 0
<1*
* 
�
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

btrace_0
ctrace_1* 

dtrace_0
etrace_1* 
�
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses
l
activation

 kernel
<bias
 m_jit_compiled_convolution_op*
jd
VARIABLE_VALUEspectral_normalization_5/kernel1layer_with_weights-1/w/.ATTRIBUTES/VARIABLE_VALUE*
* 
ke
VARIABLE_VALUEspectral_normalization_5/sn_u4layer_with_weights-1/sn_u/.ATTRIBUTES/VARIABLE_VALUE*

*0
=1
,2*

*0
=1*
* 
�
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

strace_0
ttrace_1* 

utrace_0
vtrace_1* 
�
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses
}
activation

*kernel
=bias
 ~_jit_compiled_convolution_op*
jd
VARIABLE_VALUEspectral_normalization_6/kernel1layer_with_weights-2/w/.ATTRIBUTES/VARIABLE_VALUE*
* 
ke
VARIABLE_VALUEspectral_normalization_6/sn_u4layer_with_weights-2/sn_u/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

90
:1*

90
:1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEspectral_normalization_4/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEspectral_normalization_5/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEspectral_normalization_6/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*

0
"1
,2*
.
0
1
2
3
4
5*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0*

0*
* 
* 
* 
* 
* 
* 
* 

0
;1*

0
;1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
* 

"0*

0*
* 
* 
* 
* 
* 
* 
* 

 0
<1*

 0
<1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses*
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
* 

,0*

)0*
* 
* 
* 
* 
* 
* 
* 

*0
=1*

*0
=1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses*
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
[0* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
	
l0* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
	
}0* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
serving_default_input_2Placeholder*/
_output_shapes
:���������2*
dtype0*$
shape:���������2
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2spectral_normalization_4/kernelspectral_normalization_4/biasspectral_normalization_5/kernelspectral_normalization_5/biasspectral_normalization_6/kernelspectral_normalization_6/biasdense/kernel
dense/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� */
f*R(
&__inference_signature_wrapper_95851294
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename3spectral_normalization_4/kernel/Read/ReadVariableOp1spectral_normalization_4/sn_u/Read/ReadVariableOp3spectral_normalization_5/kernel/Read/ReadVariableOp1spectral_normalization_5/sn_u/Read/ReadVariableOp3spectral_normalization_6/kernel/Read/ReadVariableOp1spectral_normalization_6/sn_u/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp1spectral_normalization_4/bias/Read/ReadVariableOp1spectral_normalization_5/bias/Read/ReadVariableOp1spectral_normalization_6/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_save_95851794
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamespectral_normalization_4/kernelspectral_normalization_4/sn_uspectral_normalization_5/kernelspectral_normalization_5/sn_uspectral_normalization_6/kernelspectral_normalization_6/sn_udense/kernel
dense/biasspectral_normalization_4/biasspectral_normalization_5/biasspectral_normalization_6/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference__traced_restore_95851837��
�
�
9__inference_discriminator_ensemble_layer_call_fn_95851342

inputs!
unknown:@
	unknown_0:@
	unknown_1:@#
	unknown_2:	@ 
	unknown_3: 
	unknown_4: #
	unknown_5: 
	unknown_6:
	unknown_7:
	unknown_8:	�
	unknown_9:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
	
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_discriminator_ensemble_layer_call_and_return_conditional_losses_95851163o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:���������2: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
V__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_95850826

inputsA
'conv2d_6_conv2d_readvariableop_resource:@6
(conv2d_6_biasadd_readvariableop_resource:@
identity��conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2@*
paddingSAME*
strides
�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2@w
conv2d_6/leaky_re_lu/LeakyRelu	LeakyReluconv2d_6/BiasAdd:output:0*/
_output_shapes
:���������2@�
IdentityIdentity,conv2d_6/leaky_re_lu/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������2@�
NoOpNoOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������2: : 2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������2
 
_user_specified_nameinputs
�
a
E__inference_flatten_layer_call_and_return_conditional_losses_95850872

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������2:W S
/
_output_shapes
:���������2
 
_user_specified_nameinputs
�7
�
V__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_95850978

inputs9
reshape_readvariableop_resource: C
1spectral_normalize_matmul_readvariableop_resource:6
(conv2d_8_biasadd_readvariableop_resource:
identity��Reshape/ReadVariableOp�conv2d_8/BiasAdd/ReadVariableOp�conv2d_8/Conv2D/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
: *
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   s
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes

:`�
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes

:`*
transpose_b(~
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes

:`v
%spectral_normalize/l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
#spectral_normalize/l2_normalize/SumSum*spectral_normalize/l2_normalize/Square:y:0.spectral_normalize/l2_normalize/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(n
)spectral_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
'spectral_normalize/l2_normalize/MaximumMaximum,spectral_normalize/l2_normalize/Sum:output:02spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:�
%spectral_normalize/l2_normalize/RsqrtRsqrt+spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:�
spectral_normalize/l2_normalizeMul#spectral_normalize/MatMul:product:0)spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes

:`�
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes

:�
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:x
'spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%spectral_normalize/l2_normalize_1/SumSum,spectral_normalize/l2_normalize_1/Square:y:00spectral_normalize/l2_normalize_1/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(p
+spectral_normalize/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
)spectral_normalize/l2_normalize_1/MaximumMaximum.spectral_normalize/l2_normalize_1/Sum:output:04spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:�
'spectral_normalize/l2_normalize_1/RsqrtRsqrt-spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:�
!spectral_normalize/l2_normalize_1Mul%spectral_normalize/MatMul_1:product:0+spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

:
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes

:`�
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes

:�
spectral_normalize/MatMul_3MatMul%spectral_normalize/MatMul_2:product:0(spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(�
#spectral_normalize/AssignVariableOpAssignVariableOp1spectral_normalize_matmul_readvariableop_resource(spectral_normalize/StopGradient:output:0)^spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
!spectral_normalize/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
: *
dtype0�
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
: y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
: �
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_8/Conv2D/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
: *
dtype0�
conv2d_8/Conv2DConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2*
paddingSAME*
strides
�
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2y
 conv2d_8/leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_8/BiasAdd:output:0*/
_output_shapes
:���������2�
IdentityIdentity.conv2d_8/leaky_re_lu_2/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������2�
NoOpNoOp^Reshape/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������2 : : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:���������2 
 
_user_specified_nameinputs
�

�
9__inference_discriminator_ensemble_layer_call_fn_95851315

inputs!
unknown:@
	unknown_0:@#
	unknown_1:	@ 
	unknown_2: #
	unknown_3: 
	unknown_4:
	unknown_5:	�
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_discriminator_ensemble_layer_call_and_return_conditional_losses_95850891o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������2: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
;__inference_spectral_normalization_5_layer_call_fn_95851586

inputs!
unknown:	@ 
	unknown_0: 
	unknown_1: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������2 *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_95851039w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������2@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������2@
 
_user_specified_nameinputs
�
F
*__inference_flatten_layer_call_fn_95851713

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_95850872a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������2:W S
/
_output_shapes
:���������2
 
_user_specified_nameinputs
�	
�
&__inference_signature_wrapper_95851294
input_2!
unknown:@
	unknown_0:@#
	unknown_1:	@ 
	unknown_2: #
	unknown_3: 
	unknown_4:
	unknown_5:	�
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference__wrapped_model_95850808o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������2: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������2
!
_user_specified_name	input_2
�
�
V__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_95850843

inputsA
'conv2d_7_conv2d_readvariableop_resource:	@ 6
(conv2d_7_biasadd_readvariableop_resource: 
identity��conv2d_7/BiasAdd/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:	@ *
dtype0�
conv2d_7/Conv2DConv2Dinputs&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2 *
paddingSAME*
strides
�
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2 y
 conv2d_7/leaky_re_lu_1/LeakyRelu	LeakyReluconv2d_7/BiasAdd:output:0*/
_output_shapes
:���������2 �
IdentityIdentity.conv2d_7/leaky_re_lu_1/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������2 �
NoOpNoOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������2@: : 2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������2@
 
_user_specified_nameinputs
�
�
;__inference_spectral_normalization_6_layer_call_fn_95851646

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_95850860w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������2 : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������2 
 
_user_specified_nameinputs
� 
�
T__inference_discriminator_ensemble_layer_call_and_return_conditional_losses_95851271
input_2;
!spectral_normalization_4_95851243:@3
!spectral_normalization_4_95851245:@/
!spectral_normalization_4_95851247:@;
!spectral_normalization_5_95851250:	@ 3
!spectral_normalization_5_95851252: /
!spectral_normalization_5_95851254: ;
!spectral_normalization_6_95851257: 3
!spectral_normalization_6_95851259:/
!spectral_normalization_6_95851261:!
dense_95851265:	�
dense_95851267:
identity��dense/StatefulPartitionedCall�0spectral_normalization_4/StatefulPartitionedCall�0spectral_normalization_5/StatefulPartitionedCall�0spectral_normalization_6/StatefulPartitionedCall�
0spectral_normalization_4/StatefulPartitionedCallStatefulPartitionedCallinput_2!spectral_normalization_4_95851243!spectral_normalization_4_95851245!spectral_normalization_4_95851247*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������2@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_95851100�
0spectral_normalization_5/StatefulPartitionedCallStatefulPartitionedCall9spectral_normalization_4/StatefulPartitionedCall:output:0!spectral_normalization_5_95851250!spectral_normalization_5_95851252!spectral_normalization_5_95851254*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������2 *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_95851039�
0spectral_normalization_6/StatefulPartitionedCallStatefulPartitionedCall9spectral_normalization_5/StatefulPartitionedCall:output:0!spectral_normalization_6_95851257!spectral_normalization_6_95851259!spectral_normalization_6_95851261*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������2*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_95850978�
flatten/PartitionedCallPartitionedCall9spectral_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_95850872�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_95851265dense_95851267*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_95850884u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall1^spectral_normalization_4/StatefulPartitionedCall1^spectral_normalization_5/StatefulPartitionedCall1^spectral_normalization_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:���������2: : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2d
0spectral_normalization_4/StatefulPartitionedCall0spectral_normalization_4/StatefulPartitionedCall2d
0spectral_normalization_5/StatefulPartitionedCall0spectral_normalization_5/StatefulPartitionedCall2d
0spectral_normalization_6/StatefulPartitionedCall0spectral_normalization_6/StatefulPartitionedCall:X T
/
_output_shapes
:���������2
!
_user_specified_name	input_2
�
�
9__inference_discriminator_ensemble_layer_call_fn_95851215
input_2!
unknown:@
	unknown_0:@
	unknown_1:@#
	unknown_2:	@ 
	unknown_3: 
	unknown_4: #
	unknown_5: 
	unknown_6:
	unknown_7:
	unknown_8:	�
	unknown_9:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
	
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_discriminator_ensemble_layer_call_and_return_conditional_losses_95851163o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:���������2: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������2
!
_user_specified_name	input_2
�7
�
V__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_95851039

inputs9
reshape_readvariableop_resource:	@ C
1spectral_normalize_matmul_readvariableop_resource: 6
(conv2d_7_biasadd_readvariableop_resource: 
identity��Reshape/ReadVariableOp�conv2d_7/BiasAdd/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:	@ *
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����    t
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	� �
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes
:	�*
transpose_b(
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	�v
%spectral_normalize/l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
#spectral_normalize/l2_normalize/SumSum*spectral_normalize/l2_normalize/Square:y:0.spectral_normalize/l2_normalize/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(n
)spectral_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
'spectral_normalize/l2_normalize/MaximumMaximum,spectral_normalize/l2_normalize/Sum:output:02spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:�
%spectral_normalize/l2_normalize/RsqrtRsqrt+spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:�
spectral_normalize/l2_normalizeMul#spectral_normalize/MatMul:product:0)spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes
:	��
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes

: �
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

: x
'spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%spectral_normalize/l2_normalize_1/SumSum,spectral_normalize/l2_normalize_1/Square:y:00spectral_normalize/l2_normalize_1/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(p
+spectral_normalize/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
)spectral_normalize/l2_normalize_1/MaximumMaximum.spectral_normalize/l2_normalize_1/Sum:output:04spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:�
'spectral_normalize/l2_normalize_1/RsqrtRsqrt-spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:�
!spectral_normalize/l2_normalize_1Mul%spectral_normalize/MatMul_1:product:0+spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

: 
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

: �
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	��
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes

: �
spectral_normalize/MatMul_3MatMul%spectral_normalize/MatMul_2:product:0(spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(�
#spectral_normalize/AssignVariableOpAssignVariableOp1spectral_normalize_matmul_readvariableop_resource(spectral_normalize/StopGradient:output:0)^spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
!spectral_normalize/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:	@ *
dtype0�
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:	@ y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   	   @       �
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:	@ �
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_7/Conv2D/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
:	@ *
dtype0�
conv2d_7/Conv2DConv2Dinputs&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2 *
paddingSAME*
strides
�
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2 y
 conv2d_7/leaky_re_lu_1/LeakyRelu	LeakyReluconv2d_7/BiasAdd:output:0*/
_output_shapes
:���������2 �
IdentityIdentity.conv2d_7/leaky_re_lu_1/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������2 �
NoOpNoOp^Reshape/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������2@: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:���������2@
 
_user_specified_nameinputs
�7
�
V__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_95851708

inputs9
reshape_readvariableop_resource: C
1spectral_normalize_matmul_readvariableop_resource:6
(conv2d_8_biasadd_readvariableop_resource:
identity��Reshape/ReadVariableOp�conv2d_8/BiasAdd/ReadVariableOp�conv2d_8/Conv2D/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
: *
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   s
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes

:`�
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes

:`*
transpose_b(~
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes

:`v
%spectral_normalize/l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
#spectral_normalize/l2_normalize/SumSum*spectral_normalize/l2_normalize/Square:y:0.spectral_normalize/l2_normalize/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(n
)spectral_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
'spectral_normalize/l2_normalize/MaximumMaximum,spectral_normalize/l2_normalize/Sum:output:02spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:�
%spectral_normalize/l2_normalize/RsqrtRsqrt+spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:�
spectral_normalize/l2_normalizeMul#spectral_normalize/MatMul:product:0)spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes

:`�
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes

:�
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:x
'spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%spectral_normalize/l2_normalize_1/SumSum,spectral_normalize/l2_normalize_1/Square:y:00spectral_normalize/l2_normalize_1/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(p
+spectral_normalize/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
)spectral_normalize/l2_normalize_1/MaximumMaximum.spectral_normalize/l2_normalize_1/Sum:output:04spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:�
'spectral_normalize/l2_normalize_1/RsqrtRsqrt-spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:�
!spectral_normalize/l2_normalize_1Mul%spectral_normalize/MatMul_1:product:0+spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

:
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes

:`�
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes

:�
spectral_normalize/MatMul_3MatMul%spectral_normalize/MatMul_2:product:0(spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(�
#spectral_normalize/AssignVariableOpAssignVariableOp1spectral_normalize_matmul_readvariableop_resource(spectral_normalize/StopGradient:output:0)^spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
!spectral_normalize/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
: *
dtype0�
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
: y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
: �
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_8/Conv2D/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
: *
dtype0�
conv2d_8/Conv2DConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2*
paddingSAME*
strides
�
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2y
 conv2d_8/leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_8/BiasAdd:output:0*/
_output_shapes
:���������2�
IdentityIdentity.conv2d_8/leaky_re_lu_2/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������2�
NoOpNoOp^Reshape/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������2 : : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:���������2 
 
_user_specified_nameinputs
�4
�	
T__inference_discriminator_ensemble_layer_call_and_return_conditional_losses_95851375

inputsZ
@spectral_normalization_4_conv2d_6_conv2d_readvariableop_resource:@O
Aspectral_normalization_4_conv2d_6_biasadd_readvariableop_resource:@Z
@spectral_normalization_5_conv2d_7_conv2d_readvariableop_resource:	@ O
Aspectral_normalization_5_conv2d_7_biasadd_readvariableop_resource: Z
@spectral_normalization_6_conv2d_8_conv2d_readvariableop_resource: O
Aspectral_normalization_6_conv2d_8_biasadd_readvariableop_resource:7
$dense_matmul_readvariableop_resource:	�3
%dense_biasadd_readvariableop_resource:
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�8spectral_normalization_4/conv2d_6/BiasAdd/ReadVariableOp�7spectral_normalization_4/conv2d_6/Conv2D/ReadVariableOp�8spectral_normalization_5/conv2d_7/BiasAdd/ReadVariableOp�7spectral_normalization_5/conv2d_7/Conv2D/ReadVariableOp�8spectral_normalization_6/conv2d_8/BiasAdd/ReadVariableOp�7spectral_normalization_6/conv2d_8/Conv2D/ReadVariableOp�
7spectral_normalization_4/conv2d_6/Conv2D/ReadVariableOpReadVariableOp@spectral_normalization_4_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
(spectral_normalization_4/conv2d_6/Conv2DConv2Dinputs?spectral_normalization_4/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2@*
paddingSAME*
strides
�
8spectral_normalization_4/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpAspectral_normalization_4_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
)spectral_normalization_4/conv2d_6/BiasAddBiasAdd1spectral_normalization_4/conv2d_6/Conv2D:output:0@spectral_normalization_4/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2@�
7spectral_normalization_4/conv2d_6/leaky_re_lu/LeakyRelu	LeakyRelu2spectral_normalization_4/conv2d_6/BiasAdd:output:0*/
_output_shapes
:���������2@�
7spectral_normalization_5/conv2d_7/Conv2D/ReadVariableOpReadVariableOp@spectral_normalization_5_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:	@ *
dtype0�
(spectral_normalization_5/conv2d_7/Conv2DConv2DEspectral_normalization_4/conv2d_6/leaky_re_lu/LeakyRelu:activations:0?spectral_normalization_5/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2 *
paddingSAME*
strides
�
8spectral_normalization_5/conv2d_7/BiasAdd/ReadVariableOpReadVariableOpAspectral_normalization_5_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
)spectral_normalization_5/conv2d_7/BiasAddBiasAdd1spectral_normalization_5/conv2d_7/Conv2D:output:0@spectral_normalization_5/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2 �
9spectral_normalization_5/conv2d_7/leaky_re_lu_1/LeakyRelu	LeakyRelu2spectral_normalization_5/conv2d_7/BiasAdd:output:0*/
_output_shapes
:���������2 �
7spectral_normalization_6/conv2d_8/Conv2D/ReadVariableOpReadVariableOp@spectral_normalization_6_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
(spectral_normalization_6/conv2d_8/Conv2DConv2DGspectral_normalization_5/conv2d_7/leaky_re_lu_1/LeakyRelu:activations:0?spectral_normalization_6/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2*
paddingSAME*
strides
�
8spectral_normalization_6/conv2d_8/BiasAdd/ReadVariableOpReadVariableOpAspectral_normalization_6_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
)spectral_normalization_6/conv2d_8/BiasAddBiasAdd1spectral_normalization_6/conv2d_8/Conv2D:output:0@spectral_normalization_6/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2�
9spectral_normalization_6/conv2d_8/leaky_re_lu_2/LeakyRelu	LeakyRelu2spectral_normalization_6/conv2d_8/BiasAdd:output:0*/
_output_shapes
:���������2^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten/ReshapeReshapeGspectral_normalization_6/conv2d_8/leaky_re_lu_2/LeakyRelu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:�����������
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp9^spectral_normalization_4/conv2d_6/BiasAdd/ReadVariableOp8^spectral_normalization_4/conv2d_6/Conv2D/ReadVariableOp9^spectral_normalization_5/conv2d_7/BiasAdd/ReadVariableOp8^spectral_normalization_5/conv2d_7/Conv2D/ReadVariableOp9^spectral_normalization_6/conv2d_8/BiasAdd/ReadVariableOp8^spectral_normalization_6/conv2d_8/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������2: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2t
8spectral_normalization_4/conv2d_6/BiasAdd/ReadVariableOp8spectral_normalization_4/conv2d_6/BiasAdd/ReadVariableOp2r
7spectral_normalization_4/conv2d_6/Conv2D/ReadVariableOp7spectral_normalization_4/conv2d_6/Conv2D/ReadVariableOp2t
8spectral_normalization_5/conv2d_7/BiasAdd/ReadVariableOp8spectral_normalization_5/conv2d_7/BiasAdd/ReadVariableOp2r
7spectral_normalization_5/conv2d_7/Conv2D/ReadVariableOp7spectral_normalization_5/conv2d_7/Conv2D/ReadVariableOp2t
8spectral_normalization_6/conv2d_8/BiasAdd/ReadVariableOp8spectral_normalization_6/conv2d_8/BiasAdd/ReadVariableOp2r
7spectral_normalization_6/conv2d_8/Conv2D/ReadVariableOp7spectral_normalization_6/conv2d_8/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
V__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_95850860

inputsA
'conv2d_8_conv2d_readvariableop_resource: 6
(conv2d_8_biasadd_readvariableop_resource:
identity��conv2d_8/BiasAdd/ReadVariableOp�conv2d_8/Conv2D/ReadVariableOp�
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_8/Conv2DConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2*
paddingSAME*
strides
�
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2y
 conv2d_8/leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_8/BiasAdd:output:0*/
_output_shapes
:���������2�
IdentityIdentity.conv2d_8/leaky_re_lu_2/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������2�
NoOpNoOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������2 : : 2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������2 
 
_user_specified_nameinputs
�7
�
V__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_95851637

inputs9
reshape_readvariableop_resource:	@ C
1spectral_normalize_matmul_readvariableop_resource: 6
(conv2d_7_biasadd_readvariableop_resource: 
identity��Reshape/ReadVariableOp�conv2d_7/BiasAdd/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:	@ *
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����    t
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	� �
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes
:	�*
transpose_b(
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	�v
%spectral_normalize/l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
#spectral_normalize/l2_normalize/SumSum*spectral_normalize/l2_normalize/Square:y:0.spectral_normalize/l2_normalize/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(n
)spectral_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
'spectral_normalize/l2_normalize/MaximumMaximum,spectral_normalize/l2_normalize/Sum:output:02spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:�
%spectral_normalize/l2_normalize/RsqrtRsqrt+spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:�
spectral_normalize/l2_normalizeMul#spectral_normalize/MatMul:product:0)spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes
:	��
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes

: �
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

: x
'spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%spectral_normalize/l2_normalize_1/SumSum,spectral_normalize/l2_normalize_1/Square:y:00spectral_normalize/l2_normalize_1/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(p
+spectral_normalize/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
)spectral_normalize/l2_normalize_1/MaximumMaximum.spectral_normalize/l2_normalize_1/Sum:output:04spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:�
'spectral_normalize/l2_normalize_1/RsqrtRsqrt-spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:�
!spectral_normalize/l2_normalize_1Mul%spectral_normalize/MatMul_1:product:0+spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

: 
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

: �
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	��
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes

: �
spectral_normalize/MatMul_3MatMul%spectral_normalize/MatMul_2:product:0(spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(�
#spectral_normalize/AssignVariableOpAssignVariableOp1spectral_normalize_matmul_readvariableop_resource(spectral_normalize/StopGradient:output:0)^spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
!spectral_normalize/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:	@ *
dtype0�
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:	@ y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   	   @       �
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:	@ �
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_7/Conv2D/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
:	@ *
dtype0�
conv2d_7/Conv2DConv2Dinputs&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2 *
paddingSAME*
strides
�
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2 y
 conv2d_7/leaky_re_lu_1/LeakyRelu	LeakyReluconv2d_7/BiasAdd:output:0*/
_output_shapes
:���������2 �
IdentityIdentity.conv2d_7/leaky_re_lu_1/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������2 �
NoOpNoOp^Reshape/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������2@: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:���������2@
 
_user_specified_nameinputs
�
�
;__inference_spectral_normalization_6_layer_call_fn_95851657

inputs!
unknown: 
	unknown_0:
	unknown_1:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������2*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_95850978w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������2 : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������2 
 
_user_specified_nameinputs
�
�
;__inference_spectral_normalization_4_layer_call_fn_95851504

inputs!
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������2@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_95850826w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������2: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
;__inference_spectral_normalization_4_layer_call_fn_95851515

inputs!
unknown:@
	unknown_0:@
	unknown_1:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������2@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_95851100w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������2
 
_user_specified_nameinputs
�

�
9__inference_discriminator_ensemble_layer_call_fn_95850910
input_2!
unknown:@
	unknown_0:@#
	unknown_1:	@ 
	unknown_2: #
	unknown_3: 
	unknown_4:
	unknown_5:	�
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_discriminator_ensemble_layer_call_and_return_conditional_losses_95850891o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������2: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������2
!
_user_specified_name	input_2
�#
�
!__inference__traced_save_95851794
file_prefix>
:savev2_spectral_normalization_4_kernel_read_readvariableop<
8savev2_spectral_normalization_4_sn_u_read_readvariableop>
:savev2_spectral_normalization_5_kernel_read_readvariableop<
8savev2_spectral_normalization_5_sn_u_read_readvariableop>
:savev2_spectral_normalization_6_kernel_read_readvariableop<
8savev2_spectral_normalization_6_sn_u_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop<
8savev2_spectral_normalization_4_bias_read_readvariableop<
8savev2_spectral_normalization_5_bias_read_readvariableop<
8savev2_spectral_normalization_6_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/sn_u/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-1/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/sn_u/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-2/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/sn_u/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0:savev2_spectral_normalization_4_kernel_read_readvariableop8savev2_spectral_normalization_4_sn_u_read_readvariableop:savev2_spectral_normalization_5_kernel_read_readvariableop8savev2_spectral_normalization_5_sn_u_read_readvariableop:savev2_spectral_normalization_6_kernel_read_readvariableop8savev2_spectral_normalization_6_sn_u_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop8savev2_spectral_normalization_4_bias_read_readvariableop8savev2_spectral_normalization_5_bias_read_readvariableop8savev2_spectral_normalization_6_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes}
{: :@:@:	@ : : ::	�::@: :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@:$ 

_output_shapes

:@:,(
&
_output_shapes
:	@ :$ 

_output_shapes

: :,(
&
_output_shapes
: :$ 

_output_shapes

::%!

_output_shapes
:	�: 

_output_shapes
:: 	

_output_shapes
:@: 


_output_shapes
: : 

_output_shapes
::

_output_shapes
: 
�
a
E__inference_flatten_layer_call_and_return_conditional_losses_95851719

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������2:W S
/
_output_shapes
:���������2
 
_user_specified_nameinputs
��
�
T__inference_discriminator_ensemble_layer_call_and_return_conditional_losses_95851495

inputsR
8spectral_normalization_4_reshape_readvariableop_resource:@\
Jspectral_normalization_4_spectral_normalize_matmul_readvariableop_resource:@O
Aspectral_normalization_4_conv2d_6_biasadd_readvariableop_resource:@R
8spectral_normalization_5_reshape_readvariableop_resource:	@ \
Jspectral_normalization_5_spectral_normalize_matmul_readvariableop_resource: O
Aspectral_normalization_5_conv2d_7_biasadd_readvariableop_resource: R
8spectral_normalization_6_reshape_readvariableop_resource: \
Jspectral_normalization_6_spectral_normalize_matmul_readvariableop_resource:O
Aspectral_normalization_6_conv2d_8_biasadd_readvariableop_resource:7
$dense_matmul_readvariableop_resource:	�3
%dense_biasadd_readvariableop_resource:
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�/spectral_normalization_4/Reshape/ReadVariableOp�8spectral_normalization_4/conv2d_6/BiasAdd/ReadVariableOp�7spectral_normalization_4/conv2d_6/Conv2D/ReadVariableOp�<spectral_normalization_4/spectral_normalize/AssignVariableOp�>spectral_normalization_4/spectral_normalize/AssignVariableOp_1�Aspectral_normalization_4/spectral_normalize/MatMul/ReadVariableOp�:spectral_normalization_4/spectral_normalize/ReadVariableOp�/spectral_normalization_5/Reshape/ReadVariableOp�8spectral_normalization_5/conv2d_7/BiasAdd/ReadVariableOp�7spectral_normalization_5/conv2d_7/Conv2D/ReadVariableOp�<spectral_normalization_5/spectral_normalize/AssignVariableOp�>spectral_normalization_5/spectral_normalize/AssignVariableOp_1�Aspectral_normalization_5/spectral_normalize/MatMul/ReadVariableOp�:spectral_normalization_5/spectral_normalize/ReadVariableOp�/spectral_normalization_6/Reshape/ReadVariableOp�8spectral_normalization_6/conv2d_8/BiasAdd/ReadVariableOp�7spectral_normalization_6/conv2d_8/Conv2D/ReadVariableOp�<spectral_normalization_6/spectral_normalize/AssignVariableOp�>spectral_normalization_6/spectral_normalize/AssignVariableOp_1�Aspectral_normalization_6/spectral_normalize/MatMul/ReadVariableOp�:spectral_normalization_6/spectral_normalize/ReadVariableOp�
/spectral_normalization_4/Reshape/ReadVariableOpReadVariableOp8spectral_normalization_4_reshape_readvariableop_resource*&
_output_shapes
:@*
dtype0w
&spectral_normalization_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
 spectral_normalization_4/ReshapeReshape7spectral_normalization_4/Reshape/ReadVariableOp:value:0/spectral_normalization_4/Reshape/shape:output:0*
T0*
_output_shapes

:H@�
Aspectral_normalization_4/spectral_normalize/MatMul/ReadVariableOpReadVariableOpJspectral_normalization_4_spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
2spectral_normalization_4/spectral_normalize/MatMulMatMulIspectral_normalization_4/spectral_normalize/MatMul/ReadVariableOp:value:0)spectral_normalization_4/Reshape:output:0*
T0*
_output_shapes

:H*
transpose_b(�
?spectral_normalization_4/spectral_normalize/l2_normalize/SquareSquare<spectral_normalization_4/spectral_normalize/MatMul:product:0*
T0*
_output_shapes

:H�
>spectral_normalization_4/spectral_normalize/l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
<spectral_normalization_4/spectral_normalize/l2_normalize/SumSumCspectral_normalization_4/spectral_normalize/l2_normalize/Square:y:0Gspectral_normalization_4/spectral_normalize/l2_normalize/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(�
Bspectral_normalization_4/spectral_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
@spectral_normalization_4/spectral_normalize/l2_normalize/MaximumMaximumEspectral_normalization_4/spectral_normalize/l2_normalize/Sum:output:0Kspectral_normalization_4/spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:�
>spectral_normalization_4/spectral_normalize/l2_normalize/RsqrtRsqrtDspectral_normalization_4/spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:�
8spectral_normalization_4/spectral_normalize/l2_normalizeMul<spectral_normalization_4/spectral_normalize/MatMul:product:0Bspectral_normalization_4/spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes

:H�
4spectral_normalization_4/spectral_normalize/MatMul_1MatMul<spectral_normalization_4/spectral_normalize/l2_normalize:z:0)spectral_normalization_4/Reshape:output:0*
T0*
_output_shapes

:@�
Aspectral_normalization_4/spectral_normalize/l2_normalize_1/SquareSquare>spectral_normalization_4/spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:@�
@spectral_normalization_4/spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
>spectral_normalization_4/spectral_normalize/l2_normalize_1/SumSumEspectral_normalization_4/spectral_normalize/l2_normalize_1/Square:y:0Ispectral_normalization_4/spectral_normalize/l2_normalize_1/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(�
Dspectral_normalization_4/spectral_normalize/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
Bspectral_normalization_4/spectral_normalize/l2_normalize_1/MaximumMaximumGspectral_normalization_4/spectral_normalize/l2_normalize_1/Sum:output:0Mspectral_normalization_4/spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:�
@spectral_normalization_4/spectral_normalize/l2_normalize_1/RsqrtRsqrtFspectral_normalization_4/spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:�
:spectral_normalization_4/spectral_normalize/l2_normalize_1Mul>spectral_normalization_4/spectral_normalize/MatMul_1:product:0Dspectral_normalization_4/spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

:@�
8spectral_normalization_4/spectral_normalize/StopGradientStopGradient>spectral_normalization_4/spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:@�
:spectral_normalization_4/spectral_normalize/StopGradient_1StopGradient<spectral_normalization_4/spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes

:H�
4spectral_normalization_4/spectral_normalize/MatMul_2MatMulCspectral_normalization_4/spectral_normalize/StopGradient_1:output:0)spectral_normalization_4/Reshape:output:0*
T0*
_output_shapes

:@�
4spectral_normalization_4/spectral_normalize/MatMul_3MatMul>spectral_normalization_4/spectral_normalize/MatMul_2:product:0Aspectral_normalization_4/spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(�
<spectral_normalization_4/spectral_normalize/AssignVariableOpAssignVariableOpJspectral_normalization_4_spectral_normalize_matmul_readvariableop_resourceAspectral_normalization_4/spectral_normalize/StopGradient:output:0B^spectral_normalization_4/spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
:spectral_normalization_4/spectral_normalize/ReadVariableOpReadVariableOp8spectral_normalization_4_reshape_readvariableop_resource*&
_output_shapes
:@*
dtype0�
3spectral_normalization_4/spectral_normalize/truedivRealDivBspectral_normalization_4/spectral_normalize/ReadVariableOp:value:0>spectral_normalization_4/spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:@�
9spectral_normalization_4/spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
3spectral_normalization_4/spectral_normalize/ReshapeReshape7spectral_normalization_4/spectral_normalize/truediv:z:0Bspectral_normalization_4/spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:@�
>spectral_normalization_4/spectral_normalize/AssignVariableOp_1AssignVariableOp8spectral_normalization_4_reshape_readvariableop_resource<spectral_normalization_4/spectral_normalize/Reshape:output:00^spectral_normalization_4/Reshape/ReadVariableOp;^spectral_normalization_4/spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
7spectral_normalization_4/conv2d_6/Conv2D/ReadVariableOpReadVariableOp8spectral_normalization_4_reshape_readvariableop_resource?^spectral_normalization_4/spectral_normalize/AssignVariableOp_1*&
_output_shapes
:@*
dtype0�
(spectral_normalization_4/conv2d_6/Conv2DConv2Dinputs?spectral_normalization_4/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2@*
paddingSAME*
strides
�
8spectral_normalization_4/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpAspectral_normalization_4_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
)spectral_normalization_4/conv2d_6/BiasAddBiasAdd1spectral_normalization_4/conv2d_6/Conv2D:output:0@spectral_normalization_4/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2@�
7spectral_normalization_4/conv2d_6/leaky_re_lu/LeakyRelu	LeakyRelu2spectral_normalization_4/conv2d_6/BiasAdd:output:0*/
_output_shapes
:���������2@�
/spectral_normalization_5/Reshape/ReadVariableOpReadVariableOp8spectral_normalization_5_reshape_readvariableop_resource*&
_output_shapes
:	@ *
dtype0w
&spectral_normalization_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
 spectral_normalization_5/ReshapeReshape7spectral_normalization_5/Reshape/ReadVariableOp:value:0/spectral_normalization_5/Reshape/shape:output:0*
T0*
_output_shapes
:	� �
Aspectral_normalization_5/spectral_normalize/MatMul/ReadVariableOpReadVariableOpJspectral_normalization_5_spectral_normalize_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
2spectral_normalization_5/spectral_normalize/MatMulMatMulIspectral_normalization_5/spectral_normalize/MatMul/ReadVariableOp:value:0)spectral_normalization_5/Reshape:output:0*
T0*
_output_shapes
:	�*
transpose_b(�
?spectral_normalization_5/spectral_normalize/l2_normalize/SquareSquare<spectral_normalization_5/spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	��
>spectral_normalization_5/spectral_normalize/l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
<spectral_normalization_5/spectral_normalize/l2_normalize/SumSumCspectral_normalization_5/spectral_normalize/l2_normalize/Square:y:0Gspectral_normalization_5/spectral_normalize/l2_normalize/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(�
Bspectral_normalization_5/spectral_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
@spectral_normalization_5/spectral_normalize/l2_normalize/MaximumMaximumEspectral_normalization_5/spectral_normalize/l2_normalize/Sum:output:0Kspectral_normalization_5/spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:�
>spectral_normalization_5/spectral_normalize/l2_normalize/RsqrtRsqrtDspectral_normalization_5/spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:�
8spectral_normalization_5/spectral_normalize/l2_normalizeMul<spectral_normalization_5/spectral_normalize/MatMul:product:0Bspectral_normalization_5/spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes
:	��
4spectral_normalization_5/spectral_normalize/MatMul_1MatMul<spectral_normalization_5/spectral_normalize/l2_normalize:z:0)spectral_normalization_5/Reshape:output:0*
T0*
_output_shapes

: �
Aspectral_normalization_5/spectral_normalize/l2_normalize_1/SquareSquare>spectral_normalization_5/spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

: �
@spectral_normalization_5/spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
>spectral_normalization_5/spectral_normalize/l2_normalize_1/SumSumEspectral_normalization_5/spectral_normalize/l2_normalize_1/Square:y:0Ispectral_normalization_5/spectral_normalize/l2_normalize_1/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(�
Dspectral_normalization_5/spectral_normalize/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
Bspectral_normalization_5/spectral_normalize/l2_normalize_1/MaximumMaximumGspectral_normalization_5/spectral_normalize/l2_normalize_1/Sum:output:0Mspectral_normalization_5/spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:�
@spectral_normalization_5/spectral_normalize/l2_normalize_1/RsqrtRsqrtFspectral_normalization_5/spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:�
:spectral_normalization_5/spectral_normalize/l2_normalize_1Mul>spectral_normalization_5/spectral_normalize/MatMul_1:product:0Dspectral_normalization_5/spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

: �
8spectral_normalization_5/spectral_normalize/StopGradientStopGradient>spectral_normalization_5/spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

: �
:spectral_normalization_5/spectral_normalize/StopGradient_1StopGradient<spectral_normalization_5/spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	��
4spectral_normalization_5/spectral_normalize/MatMul_2MatMulCspectral_normalization_5/spectral_normalize/StopGradient_1:output:0)spectral_normalization_5/Reshape:output:0*
T0*
_output_shapes

: �
4spectral_normalization_5/spectral_normalize/MatMul_3MatMul>spectral_normalization_5/spectral_normalize/MatMul_2:product:0Aspectral_normalization_5/spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(�
<spectral_normalization_5/spectral_normalize/AssignVariableOpAssignVariableOpJspectral_normalization_5_spectral_normalize_matmul_readvariableop_resourceAspectral_normalization_5/spectral_normalize/StopGradient:output:0B^spectral_normalization_5/spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
:spectral_normalization_5/spectral_normalize/ReadVariableOpReadVariableOp8spectral_normalization_5_reshape_readvariableop_resource*&
_output_shapes
:	@ *
dtype0�
3spectral_normalization_5/spectral_normalize/truedivRealDivBspectral_normalization_5/spectral_normalize/ReadVariableOp:value:0>spectral_normalization_5/spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:	@ �
9spectral_normalization_5/spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   	   @       �
3spectral_normalization_5/spectral_normalize/ReshapeReshape7spectral_normalization_5/spectral_normalize/truediv:z:0Bspectral_normalization_5/spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:	@ �
>spectral_normalization_5/spectral_normalize/AssignVariableOp_1AssignVariableOp8spectral_normalization_5_reshape_readvariableop_resource<spectral_normalization_5/spectral_normalize/Reshape:output:00^spectral_normalization_5/Reshape/ReadVariableOp;^spectral_normalization_5/spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
7spectral_normalization_5/conv2d_7/Conv2D/ReadVariableOpReadVariableOp8spectral_normalization_5_reshape_readvariableop_resource?^spectral_normalization_5/spectral_normalize/AssignVariableOp_1*&
_output_shapes
:	@ *
dtype0�
(spectral_normalization_5/conv2d_7/Conv2DConv2DEspectral_normalization_4/conv2d_6/leaky_re_lu/LeakyRelu:activations:0?spectral_normalization_5/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2 *
paddingSAME*
strides
�
8spectral_normalization_5/conv2d_7/BiasAdd/ReadVariableOpReadVariableOpAspectral_normalization_5_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
)spectral_normalization_5/conv2d_7/BiasAddBiasAdd1spectral_normalization_5/conv2d_7/Conv2D:output:0@spectral_normalization_5/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2 �
9spectral_normalization_5/conv2d_7/leaky_re_lu_1/LeakyRelu	LeakyRelu2spectral_normalization_5/conv2d_7/BiasAdd:output:0*/
_output_shapes
:���������2 �
/spectral_normalization_6/Reshape/ReadVariableOpReadVariableOp8spectral_normalization_6_reshape_readvariableop_resource*&
_output_shapes
: *
dtype0w
&spectral_normalization_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
 spectral_normalization_6/ReshapeReshape7spectral_normalization_6/Reshape/ReadVariableOp:value:0/spectral_normalization_6/Reshape/shape:output:0*
T0*
_output_shapes

:`�
Aspectral_normalization_6/spectral_normalize/MatMul/ReadVariableOpReadVariableOpJspectral_normalization_6_spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
2spectral_normalization_6/spectral_normalize/MatMulMatMulIspectral_normalization_6/spectral_normalize/MatMul/ReadVariableOp:value:0)spectral_normalization_6/Reshape:output:0*
T0*
_output_shapes

:`*
transpose_b(�
?spectral_normalization_6/spectral_normalize/l2_normalize/SquareSquare<spectral_normalization_6/spectral_normalize/MatMul:product:0*
T0*
_output_shapes

:`�
>spectral_normalization_6/spectral_normalize/l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
<spectral_normalization_6/spectral_normalize/l2_normalize/SumSumCspectral_normalization_6/spectral_normalize/l2_normalize/Square:y:0Gspectral_normalization_6/spectral_normalize/l2_normalize/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(�
Bspectral_normalization_6/spectral_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
@spectral_normalization_6/spectral_normalize/l2_normalize/MaximumMaximumEspectral_normalization_6/spectral_normalize/l2_normalize/Sum:output:0Kspectral_normalization_6/spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:�
>spectral_normalization_6/spectral_normalize/l2_normalize/RsqrtRsqrtDspectral_normalization_6/spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:�
8spectral_normalization_6/spectral_normalize/l2_normalizeMul<spectral_normalization_6/spectral_normalize/MatMul:product:0Bspectral_normalization_6/spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes

:`�
4spectral_normalization_6/spectral_normalize/MatMul_1MatMul<spectral_normalization_6/spectral_normalize/l2_normalize:z:0)spectral_normalization_6/Reshape:output:0*
T0*
_output_shapes

:�
Aspectral_normalization_6/spectral_normalize/l2_normalize_1/SquareSquare>spectral_normalization_6/spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:�
@spectral_normalization_6/spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
>spectral_normalization_6/spectral_normalize/l2_normalize_1/SumSumEspectral_normalization_6/spectral_normalize/l2_normalize_1/Square:y:0Ispectral_normalization_6/spectral_normalize/l2_normalize_1/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(�
Dspectral_normalization_6/spectral_normalize/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
Bspectral_normalization_6/spectral_normalize/l2_normalize_1/MaximumMaximumGspectral_normalization_6/spectral_normalize/l2_normalize_1/Sum:output:0Mspectral_normalization_6/spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:�
@spectral_normalization_6/spectral_normalize/l2_normalize_1/RsqrtRsqrtFspectral_normalization_6/spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:�
:spectral_normalization_6/spectral_normalize/l2_normalize_1Mul>spectral_normalization_6/spectral_normalize/MatMul_1:product:0Dspectral_normalization_6/spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

:�
8spectral_normalization_6/spectral_normalize/StopGradientStopGradient>spectral_normalization_6/spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:�
:spectral_normalization_6/spectral_normalize/StopGradient_1StopGradient<spectral_normalization_6/spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes

:`�
4spectral_normalization_6/spectral_normalize/MatMul_2MatMulCspectral_normalization_6/spectral_normalize/StopGradient_1:output:0)spectral_normalization_6/Reshape:output:0*
T0*
_output_shapes

:�
4spectral_normalization_6/spectral_normalize/MatMul_3MatMul>spectral_normalization_6/spectral_normalize/MatMul_2:product:0Aspectral_normalization_6/spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(�
<spectral_normalization_6/spectral_normalize/AssignVariableOpAssignVariableOpJspectral_normalization_6_spectral_normalize_matmul_readvariableop_resourceAspectral_normalization_6/spectral_normalize/StopGradient:output:0B^spectral_normalization_6/spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
:spectral_normalization_6/spectral_normalize/ReadVariableOpReadVariableOp8spectral_normalization_6_reshape_readvariableop_resource*&
_output_shapes
: *
dtype0�
3spectral_normalization_6/spectral_normalize/truedivRealDivBspectral_normalization_6/spectral_normalize/ReadVariableOp:value:0>spectral_normalization_6/spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
: �
9spectral_normalization_6/spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
3spectral_normalization_6/spectral_normalize/ReshapeReshape7spectral_normalization_6/spectral_normalize/truediv:z:0Bspectral_normalization_6/spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
: �
>spectral_normalization_6/spectral_normalize/AssignVariableOp_1AssignVariableOp8spectral_normalization_6_reshape_readvariableop_resource<spectral_normalization_6/spectral_normalize/Reshape:output:00^spectral_normalization_6/Reshape/ReadVariableOp;^spectral_normalization_6/spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
7spectral_normalization_6/conv2d_8/Conv2D/ReadVariableOpReadVariableOp8spectral_normalization_6_reshape_readvariableop_resource?^spectral_normalization_6/spectral_normalize/AssignVariableOp_1*&
_output_shapes
: *
dtype0�
(spectral_normalization_6/conv2d_8/Conv2DConv2DGspectral_normalization_5/conv2d_7/leaky_re_lu_1/LeakyRelu:activations:0?spectral_normalization_6/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2*
paddingSAME*
strides
�
8spectral_normalization_6/conv2d_8/BiasAdd/ReadVariableOpReadVariableOpAspectral_normalization_6_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
)spectral_normalization_6/conv2d_8/BiasAddBiasAdd1spectral_normalization_6/conv2d_8/Conv2D:output:0@spectral_normalization_6/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2�
9spectral_normalization_6/conv2d_8/leaky_re_lu_2/LeakyRelu	LeakyRelu2spectral_normalization_6/conv2d_8/BiasAdd:output:0*/
_output_shapes
:���������2^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten/ReshapeReshapeGspectral_normalization_6/conv2d_8/leaky_re_lu_2/LeakyRelu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:�����������
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������

NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp0^spectral_normalization_4/Reshape/ReadVariableOp9^spectral_normalization_4/conv2d_6/BiasAdd/ReadVariableOp8^spectral_normalization_4/conv2d_6/Conv2D/ReadVariableOp=^spectral_normalization_4/spectral_normalize/AssignVariableOp?^spectral_normalization_4/spectral_normalize/AssignVariableOp_1B^spectral_normalization_4/spectral_normalize/MatMul/ReadVariableOp;^spectral_normalization_4/spectral_normalize/ReadVariableOp0^spectral_normalization_5/Reshape/ReadVariableOp9^spectral_normalization_5/conv2d_7/BiasAdd/ReadVariableOp8^spectral_normalization_5/conv2d_7/Conv2D/ReadVariableOp=^spectral_normalization_5/spectral_normalize/AssignVariableOp?^spectral_normalization_5/spectral_normalize/AssignVariableOp_1B^spectral_normalization_5/spectral_normalize/MatMul/ReadVariableOp;^spectral_normalization_5/spectral_normalize/ReadVariableOp0^spectral_normalization_6/Reshape/ReadVariableOp9^spectral_normalization_6/conv2d_8/BiasAdd/ReadVariableOp8^spectral_normalization_6/conv2d_8/Conv2D/ReadVariableOp=^spectral_normalization_6/spectral_normalize/AssignVariableOp?^spectral_normalization_6/spectral_normalize/AssignVariableOp_1B^spectral_normalization_6/spectral_normalize/MatMul/ReadVariableOp;^spectral_normalization_6/spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:���������2: : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2b
/spectral_normalization_4/Reshape/ReadVariableOp/spectral_normalization_4/Reshape/ReadVariableOp2t
8spectral_normalization_4/conv2d_6/BiasAdd/ReadVariableOp8spectral_normalization_4/conv2d_6/BiasAdd/ReadVariableOp2r
7spectral_normalization_4/conv2d_6/Conv2D/ReadVariableOp7spectral_normalization_4/conv2d_6/Conv2D/ReadVariableOp2|
<spectral_normalization_4/spectral_normalize/AssignVariableOp<spectral_normalization_4/spectral_normalize/AssignVariableOp2�
>spectral_normalization_4/spectral_normalize/AssignVariableOp_1>spectral_normalization_4/spectral_normalize/AssignVariableOp_12�
Aspectral_normalization_4/spectral_normalize/MatMul/ReadVariableOpAspectral_normalization_4/spectral_normalize/MatMul/ReadVariableOp2x
:spectral_normalization_4/spectral_normalize/ReadVariableOp:spectral_normalization_4/spectral_normalize/ReadVariableOp2b
/spectral_normalization_5/Reshape/ReadVariableOp/spectral_normalization_5/Reshape/ReadVariableOp2t
8spectral_normalization_5/conv2d_7/BiasAdd/ReadVariableOp8spectral_normalization_5/conv2d_7/BiasAdd/ReadVariableOp2r
7spectral_normalization_5/conv2d_7/Conv2D/ReadVariableOp7spectral_normalization_5/conv2d_7/Conv2D/ReadVariableOp2|
<spectral_normalization_5/spectral_normalize/AssignVariableOp<spectral_normalization_5/spectral_normalize/AssignVariableOp2�
>spectral_normalization_5/spectral_normalize/AssignVariableOp_1>spectral_normalization_5/spectral_normalize/AssignVariableOp_12�
Aspectral_normalization_5/spectral_normalize/MatMul/ReadVariableOpAspectral_normalization_5/spectral_normalize/MatMul/ReadVariableOp2x
:spectral_normalization_5/spectral_normalize/ReadVariableOp:spectral_normalization_5/spectral_normalize/ReadVariableOp2b
/spectral_normalization_6/Reshape/ReadVariableOp/spectral_normalization_6/Reshape/ReadVariableOp2t
8spectral_normalization_6/conv2d_8/BiasAdd/ReadVariableOp8spectral_normalization_6/conv2d_8/BiasAdd/ReadVariableOp2r
7spectral_normalization_6/conv2d_8/Conv2D/ReadVariableOp7spectral_normalization_6/conv2d_8/Conv2D/ReadVariableOp2|
<spectral_normalization_6/spectral_normalize/AssignVariableOp<spectral_normalization_6/spectral_normalize/AssignVariableOp2�
>spectral_normalization_6/spectral_normalize/AssignVariableOp_1>spectral_normalization_6/spectral_normalize/AssignVariableOp_12�
Aspectral_normalization_6/spectral_normalize/MatMul/ReadVariableOpAspectral_normalization_6/spectral_normalize/MatMul/ReadVariableOp2x
:spectral_normalization_6/spectral_normalize/ReadVariableOp:spectral_normalization_6/spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:���������2
 
_user_specified_nameinputs
�7
�
V__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_95851100

inputs9
reshape_readvariableop_resource:@C
1spectral_normalize_matmul_readvariableop_resource:@6
(conv2d_6_biasadd_readvariableop_resource:@
identity��Reshape/ReadVariableOp�conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:@*
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   s
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes

:H@�
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes

:H*
transpose_b(~
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes

:Hv
%spectral_normalize/l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
#spectral_normalize/l2_normalize/SumSum*spectral_normalize/l2_normalize/Square:y:0.spectral_normalize/l2_normalize/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(n
)spectral_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
'spectral_normalize/l2_normalize/MaximumMaximum,spectral_normalize/l2_normalize/Sum:output:02spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:�
%spectral_normalize/l2_normalize/RsqrtRsqrt+spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:�
spectral_normalize/l2_normalizeMul#spectral_normalize/MatMul:product:0)spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes

:H�
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes

:@�
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:@x
'spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%spectral_normalize/l2_normalize_1/SumSum,spectral_normalize/l2_normalize_1/Square:y:00spectral_normalize/l2_normalize_1/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(p
+spectral_normalize/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
)spectral_normalize/l2_normalize_1/MaximumMaximum.spectral_normalize/l2_normalize_1/Sum:output:04spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:�
'spectral_normalize/l2_normalize_1/RsqrtRsqrt-spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:�
!spectral_normalize/l2_normalize_1Mul%spectral_normalize/MatMul_1:product:0+spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

:@
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:@
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes

:H�
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes

:@�
spectral_normalize/MatMul_3MatMul%spectral_normalize/MatMul_2:product:0(spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(�
#spectral_normalize/AssignVariableOpAssignVariableOp1spectral_normalize_matmul_readvariableop_resource(spectral_normalize/StopGradient:output:0)^spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
!spectral_normalize/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:@*
dtype0�
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:@y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:@�
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_6/Conv2D/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
:@*
dtype0�
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2@*
paddingSAME*
strides
�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2@w
conv2d_6/leaky_re_lu/LeakyRelu	LeakyReluconv2d_6/BiasAdd:output:0*/
_output_shapes
:���������2@�
IdentityIdentity,conv2d_6/leaky_re_lu/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������2@�
NoOpNoOp^Reshape/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������2: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
V__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_95851526

inputsA
'conv2d_6_conv2d_readvariableop_resource:@6
(conv2d_6_biasadd_readvariableop_resource:@
identity��conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2@*
paddingSAME*
strides
�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2@w
conv2d_6/leaky_re_lu/LeakyRelu	LeakyReluconv2d_6/BiasAdd:output:0*/
_output_shapes
:���������2@�
IdentityIdentity,conv2d_6/leaky_re_lu/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������2@�
NoOpNoOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������2: : 2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������2
 
_user_specified_nameinputs
�0
�
$__inference__traced_restore_95851837
file_prefixJ
0assignvariableop_spectral_normalization_4_kernel:@B
0assignvariableop_1_spectral_normalization_4_sn_u:@L
2assignvariableop_2_spectral_normalization_5_kernel:	@ B
0assignvariableop_3_spectral_normalization_5_sn_u: L
2assignvariableop_4_spectral_normalization_6_kernel: B
0assignvariableop_5_spectral_normalization_6_sn_u:2
assignvariableop_6_dense_kernel:	�+
assignvariableop_7_dense_bias:>
0assignvariableop_8_spectral_normalization_4_bias:@>
0assignvariableop_9_spectral_normalization_5_bias: ?
1assignvariableop_10_spectral_normalization_6_bias:
identity_12��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/sn_u/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-1/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/sn_u/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-2/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/sn_u/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*D
_output_shapes2
0::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp0assignvariableop_spectral_normalization_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp0assignvariableop_1_spectral_normalization_4_sn_uIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp2assignvariableop_2_spectral_normalization_5_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp0assignvariableop_3_spectral_normalization_5_sn_uIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp2assignvariableop_4_spectral_normalization_6_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp0assignvariableop_5_spectral_normalization_6_sn_uIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp0assignvariableop_8_spectral_normalization_4_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp0assignvariableop_9_spectral_normalization_5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp1assignvariableop_10_spectral_normalization_6_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_11Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_12IdentityIdentity_11:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_12Identity_12:output:0*+
_input_shapes
: : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�7
�
V__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_95851566

inputs9
reshape_readvariableop_resource:@C
1spectral_normalize_matmul_readvariableop_resource:@6
(conv2d_6_biasadd_readvariableop_resource:@
identity��Reshape/ReadVariableOp�conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:@*
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   s
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes

:H@�
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes

:H*
transpose_b(~
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes

:Hv
%spectral_normalize/l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
#spectral_normalize/l2_normalize/SumSum*spectral_normalize/l2_normalize/Square:y:0.spectral_normalize/l2_normalize/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(n
)spectral_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
'spectral_normalize/l2_normalize/MaximumMaximum,spectral_normalize/l2_normalize/Sum:output:02spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:�
%spectral_normalize/l2_normalize/RsqrtRsqrt+spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:�
spectral_normalize/l2_normalizeMul#spectral_normalize/MatMul:product:0)spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes

:H�
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes

:@�
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:@x
'spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
%spectral_normalize/l2_normalize_1/SumSum,spectral_normalize/l2_normalize_1/Square:y:00spectral_normalize/l2_normalize_1/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(p
+spectral_normalize/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
)spectral_normalize/l2_normalize_1/MaximumMaximum.spectral_normalize/l2_normalize_1/Sum:output:04spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:�
'spectral_normalize/l2_normalize_1/RsqrtRsqrt-spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:�
!spectral_normalize/l2_normalize_1Mul%spectral_normalize/MatMul_1:product:0+spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

:@
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:@
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes

:H�
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes

:@�
spectral_normalize/MatMul_3MatMul%spectral_normalize/MatMul_2:product:0(spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(�
#spectral_normalize/AssignVariableOpAssignVariableOp1spectral_normalize_matmul_readvariableop_resource(spectral_normalize/StopGradient:output:0)^spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
!spectral_normalize/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:@*
dtype0�
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:@y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:@�
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_6/Conv2D/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
:@*
dtype0�
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2@*
paddingSAME*
strides
�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2@w
conv2d_6/leaky_re_lu/LeakyRelu	LeakyReluconv2d_6/BiasAdd:output:0*/
_output_shapes
:���������2@�
IdentityIdentity,conv2d_6/leaky_re_lu/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������2@�
NoOpNoOp^Reshape/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������2: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
(__inference_dense_layer_call_fn_95851728

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_95850884o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
C__inference_dense_layer_call_and_return_conditional_losses_95850884

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�D
�
#__inference__wrapped_model_95850808
input_2q
Wdiscriminator_ensemble_spectral_normalization_4_conv2d_6_conv2d_readvariableop_resource:@f
Xdiscriminator_ensemble_spectral_normalization_4_conv2d_6_biasadd_readvariableop_resource:@q
Wdiscriminator_ensemble_spectral_normalization_5_conv2d_7_conv2d_readvariableop_resource:	@ f
Xdiscriminator_ensemble_spectral_normalization_5_conv2d_7_biasadd_readvariableop_resource: q
Wdiscriminator_ensemble_spectral_normalization_6_conv2d_8_conv2d_readvariableop_resource: f
Xdiscriminator_ensemble_spectral_normalization_6_conv2d_8_biasadd_readvariableop_resource:N
;discriminator_ensemble_dense_matmul_readvariableop_resource:	�J
<discriminator_ensemble_dense_biasadd_readvariableop_resource:
identity��3discriminator_ensemble/dense/BiasAdd/ReadVariableOp�2discriminator_ensemble/dense/MatMul/ReadVariableOp�Odiscriminator_ensemble/spectral_normalization_4/conv2d_6/BiasAdd/ReadVariableOp�Ndiscriminator_ensemble/spectral_normalization_4/conv2d_6/Conv2D/ReadVariableOp�Odiscriminator_ensemble/spectral_normalization_5/conv2d_7/BiasAdd/ReadVariableOp�Ndiscriminator_ensemble/spectral_normalization_5/conv2d_7/Conv2D/ReadVariableOp�Odiscriminator_ensemble/spectral_normalization_6/conv2d_8/BiasAdd/ReadVariableOp�Ndiscriminator_ensemble/spectral_normalization_6/conv2d_8/Conv2D/ReadVariableOp�
Ndiscriminator_ensemble/spectral_normalization_4/conv2d_6/Conv2D/ReadVariableOpReadVariableOpWdiscriminator_ensemble_spectral_normalization_4_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
?discriminator_ensemble/spectral_normalization_4/conv2d_6/Conv2DConv2Dinput_2Vdiscriminator_ensemble/spectral_normalization_4/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2@*
paddingSAME*
strides
�
Odiscriminator_ensemble/spectral_normalization_4/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpXdiscriminator_ensemble_spectral_normalization_4_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
@discriminator_ensemble/spectral_normalization_4/conv2d_6/BiasAddBiasAddHdiscriminator_ensemble/spectral_normalization_4/conv2d_6/Conv2D:output:0Wdiscriminator_ensemble/spectral_normalization_4/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2@�
Ndiscriminator_ensemble/spectral_normalization_4/conv2d_6/leaky_re_lu/LeakyRelu	LeakyReluIdiscriminator_ensemble/spectral_normalization_4/conv2d_6/BiasAdd:output:0*/
_output_shapes
:���������2@�
Ndiscriminator_ensemble/spectral_normalization_5/conv2d_7/Conv2D/ReadVariableOpReadVariableOpWdiscriminator_ensemble_spectral_normalization_5_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:	@ *
dtype0�
?discriminator_ensemble/spectral_normalization_5/conv2d_7/Conv2DConv2D\discriminator_ensemble/spectral_normalization_4/conv2d_6/leaky_re_lu/LeakyRelu:activations:0Vdiscriminator_ensemble/spectral_normalization_5/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2 *
paddingSAME*
strides
�
Odiscriminator_ensemble/spectral_normalization_5/conv2d_7/BiasAdd/ReadVariableOpReadVariableOpXdiscriminator_ensemble_spectral_normalization_5_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
@discriminator_ensemble/spectral_normalization_5/conv2d_7/BiasAddBiasAddHdiscriminator_ensemble/spectral_normalization_5/conv2d_7/Conv2D:output:0Wdiscriminator_ensemble/spectral_normalization_5/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2 �
Pdiscriminator_ensemble/spectral_normalization_5/conv2d_7/leaky_re_lu_1/LeakyRelu	LeakyReluIdiscriminator_ensemble/spectral_normalization_5/conv2d_7/BiasAdd:output:0*/
_output_shapes
:���������2 �
Ndiscriminator_ensemble/spectral_normalization_6/conv2d_8/Conv2D/ReadVariableOpReadVariableOpWdiscriminator_ensemble_spectral_normalization_6_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
?discriminator_ensemble/spectral_normalization_6/conv2d_8/Conv2DConv2D^discriminator_ensemble/spectral_normalization_5/conv2d_7/leaky_re_lu_1/LeakyRelu:activations:0Vdiscriminator_ensemble/spectral_normalization_6/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2*
paddingSAME*
strides
�
Odiscriminator_ensemble/spectral_normalization_6/conv2d_8/BiasAdd/ReadVariableOpReadVariableOpXdiscriminator_ensemble_spectral_normalization_6_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
@discriminator_ensemble/spectral_normalization_6/conv2d_8/BiasAddBiasAddHdiscriminator_ensemble/spectral_normalization_6/conv2d_8/Conv2D:output:0Wdiscriminator_ensemble/spectral_normalization_6/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2�
Pdiscriminator_ensemble/spectral_normalization_6/conv2d_8/leaky_re_lu_2/LeakyRelu	LeakyReluIdiscriminator_ensemble/spectral_normalization_6/conv2d_8/BiasAdd:output:0*/
_output_shapes
:���������2u
$discriminator_ensemble/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
&discriminator_ensemble/flatten/ReshapeReshape^discriminator_ensemble/spectral_normalization_6/conv2d_8/leaky_re_lu_2/LeakyRelu:activations:0-discriminator_ensemble/flatten/Const:output:0*
T0*(
_output_shapes
:�����������
2discriminator_ensemble/dense/MatMul/ReadVariableOpReadVariableOp;discriminator_ensemble_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
#discriminator_ensemble/dense/MatMulMatMul/discriminator_ensemble/flatten/Reshape:output:0:discriminator_ensemble/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
3discriminator_ensemble/dense/BiasAdd/ReadVariableOpReadVariableOp<discriminator_ensemble_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
$discriminator_ensemble/dense/BiasAddBiasAdd-discriminator_ensemble/dense/MatMul:product:0;discriminator_ensemble/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������|
IdentityIdentity-discriminator_ensemble/dense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp4^discriminator_ensemble/dense/BiasAdd/ReadVariableOp3^discriminator_ensemble/dense/MatMul/ReadVariableOpP^discriminator_ensemble/spectral_normalization_4/conv2d_6/BiasAdd/ReadVariableOpO^discriminator_ensemble/spectral_normalization_4/conv2d_6/Conv2D/ReadVariableOpP^discriminator_ensemble/spectral_normalization_5/conv2d_7/BiasAdd/ReadVariableOpO^discriminator_ensemble/spectral_normalization_5/conv2d_7/Conv2D/ReadVariableOpP^discriminator_ensemble/spectral_normalization_6/conv2d_8/BiasAdd/ReadVariableOpO^discriminator_ensemble/spectral_normalization_6/conv2d_8/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������2: : : : : : : : 2j
3discriminator_ensemble/dense/BiasAdd/ReadVariableOp3discriminator_ensemble/dense/BiasAdd/ReadVariableOp2h
2discriminator_ensemble/dense/MatMul/ReadVariableOp2discriminator_ensemble/dense/MatMul/ReadVariableOp2�
Odiscriminator_ensemble/spectral_normalization_4/conv2d_6/BiasAdd/ReadVariableOpOdiscriminator_ensemble/spectral_normalization_4/conv2d_6/BiasAdd/ReadVariableOp2�
Ndiscriminator_ensemble/spectral_normalization_4/conv2d_6/Conv2D/ReadVariableOpNdiscriminator_ensemble/spectral_normalization_4/conv2d_6/Conv2D/ReadVariableOp2�
Odiscriminator_ensemble/spectral_normalization_5/conv2d_7/BiasAdd/ReadVariableOpOdiscriminator_ensemble/spectral_normalization_5/conv2d_7/BiasAdd/ReadVariableOp2�
Ndiscriminator_ensemble/spectral_normalization_5/conv2d_7/Conv2D/ReadVariableOpNdiscriminator_ensemble/spectral_normalization_5/conv2d_7/Conv2D/ReadVariableOp2�
Odiscriminator_ensemble/spectral_normalization_6/conv2d_8/BiasAdd/ReadVariableOpOdiscriminator_ensemble/spectral_normalization_6/conv2d_8/BiasAdd/ReadVariableOp2�
Ndiscriminator_ensemble/spectral_normalization_6/conv2d_8/Conv2D/ReadVariableOpNdiscriminator_ensemble/spectral_normalization_6/conv2d_8/Conv2D/ReadVariableOp:X T
/
_output_shapes
:���������2
!
_user_specified_name	input_2
�
�
V__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_95851668

inputsA
'conv2d_8_conv2d_readvariableop_resource: 6
(conv2d_8_biasadd_readvariableop_resource:
identity��conv2d_8/BiasAdd/ReadVariableOp�conv2d_8/Conv2D/ReadVariableOp�
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_8/Conv2DConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2*
paddingSAME*
strides
�
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2y
 conv2d_8/leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_8/BiasAdd:output:0*/
_output_shapes
:���������2�
IdentityIdentity.conv2d_8/leaky_re_lu_2/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������2�
NoOpNoOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������2 : : 2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������2 
 
_user_specified_nameinputs
�
�
V__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_95851597

inputsA
'conv2d_7_conv2d_readvariableop_resource:	@ 6
(conv2d_7_biasadd_readvariableop_resource: 
identity��conv2d_7/BiasAdd/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:	@ *
dtype0�
conv2d_7/Conv2DConv2Dinputs&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2 *
paddingSAME*
strides
�
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2 y
 conv2d_7/leaky_re_lu_1/LeakyRelu	LeakyReluconv2d_7/BiasAdd:output:0*/
_output_shapes
:���������2 �
IdentityIdentity.conv2d_7/leaky_re_lu_1/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������2 �
NoOpNoOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������2@: : 2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������2@
 
_user_specified_nameinputs
�	
�
C__inference_dense_layer_call_and_return_conditional_losses_95851738

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
T__inference_discriminator_ensemble_layer_call_and_return_conditional_losses_95851240
input_2;
!spectral_normalization_4_95851218:@/
!spectral_normalization_4_95851220:@;
!spectral_normalization_5_95851223:	@ /
!spectral_normalization_5_95851225: ;
!spectral_normalization_6_95851228: /
!spectral_normalization_6_95851230:!
dense_95851234:	�
dense_95851236:
identity��dense/StatefulPartitionedCall�0spectral_normalization_4/StatefulPartitionedCall�0spectral_normalization_5/StatefulPartitionedCall�0spectral_normalization_6/StatefulPartitionedCall�
0spectral_normalization_4/StatefulPartitionedCallStatefulPartitionedCallinput_2!spectral_normalization_4_95851218!spectral_normalization_4_95851220*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������2@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_95850826�
0spectral_normalization_5/StatefulPartitionedCallStatefulPartitionedCall9spectral_normalization_4/StatefulPartitionedCall:output:0!spectral_normalization_5_95851223!spectral_normalization_5_95851225*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������2 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_95850843�
0spectral_normalization_6/StatefulPartitionedCallStatefulPartitionedCall9spectral_normalization_5/StatefulPartitionedCall:output:0!spectral_normalization_6_95851228!spectral_normalization_6_95851230*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_95850860�
flatten/PartitionedCallPartitionedCall9spectral_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_95850872�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_95851234dense_95851236*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_95850884u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall1^spectral_normalization_4/StatefulPartitionedCall1^spectral_normalization_5/StatefulPartitionedCall1^spectral_normalization_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������2: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2d
0spectral_normalization_4/StatefulPartitionedCall0spectral_normalization_4/StatefulPartitionedCall2d
0spectral_normalization_5/StatefulPartitionedCall0spectral_normalization_5/StatefulPartitionedCall2d
0spectral_normalization_6/StatefulPartitionedCall0spectral_normalization_6/StatefulPartitionedCall:X T
/
_output_shapes
:���������2
!
_user_specified_name	input_2
�
�
;__inference_spectral_normalization_5_layer_call_fn_95851575

inputs!
unknown:	@ 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������2 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_95850843w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������2@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������2@
 
_user_specified_nameinputs
� 
�
T__inference_discriminator_ensemble_layer_call_and_return_conditional_losses_95851163

inputs;
!spectral_normalization_4_95851135:@3
!spectral_normalization_4_95851137:@/
!spectral_normalization_4_95851139:@;
!spectral_normalization_5_95851142:	@ 3
!spectral_normalization_5_95851144: /
!spectral_normalization_5_95851146: ;
!spectral_normalization_6_95851149: 3
!spectral_normalization_6_95851151:/
!spectral_normalization_6_95851153:!
dense_95851157:	�
dense_95851159:
identity��dense/StatefulPartitionedCall�0spectral_normalization_4/StatefulPartitionedCall�0spectral_normalization_5/StatefulPartitionedCall�0spectral_normalization_6/StatefulPartitionedCall�
0spectral_normalization_4/StatefulPartitionedCallStatefulPartitionedCallinputs!spectral_normalization_4_95851135!spectral_normalization_4_95851137!spectral_normalization_4_95851139*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������2@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_95851100�
0spectral_normalization_5/StatefulPartitionedCallStatefulPartitionedCall9spectral_normalization_4/StatefulPartitionedCall:output:0!spectral_normalization_5_95851142!spectral_normalization_5_95851144!spectral_normalization_5_95851146*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������2 *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_95851039�
0spectral_normalization_6/StatefulPartitionedCallStatefulPartitionedCall9spectral_normalization_5/StatefulPartitionedCall:output:0!spectral_normalization_6_95851149!spectral_normalization_6_95851151!spectral_normalization_6_95851153*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������2*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_95850978�
flatten/PartitionedCallPartitionedCall9spectral_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_95850872�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_95851157dense_95851159*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_95850884u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall1^spectral_normalization_4/StatefulPartitionedCall1^spectral_normalization_5/StatefulPartitionedCall1^spectral_normalization_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:���������2: : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2d
0spectral_normalization_4/StatefulPartitionedCall0spectral_normalization_4/StatefulPartitionedCall2d
0spectral_normalization_5/StatefulPartitionedCall0spectral_normalization_5/StatefulPartitionedCall2d
0spectral_normalization_6/StatefulPartitionedCall0spectral_normalization_6/StatefulPartitionedCall:W S
/
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
T__inference_discriminator_ensemble_layer_call_and_return_conditional_losses_95850891

inputs;
!spectral_normalization_4_95850827:@/
!spectral_normalization_4_95850829:@;
!spectral_normalization_5_95850844:	@ /
!spectral_normalization_5_95850846: ;
!spectral_normalization_6_95850861: /
!spectral_normalization_6_95850863:!
dense_95850885:	�
dense_95850887:
identity��dense/StatefulPartitionedCall�0spectral_normalization_4/StatefulPartitionedCall�0spectral_normalization_5/StatefulPartitionedCall�0spectral_normalization_6/StatefulPartitionedCall�
0spectral_normalization_4/StatefulPartitionedCallStatefulPartitionedCallinputs!spectral_normalization_4_95850827!spectral_normalization_4_95850829*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������2@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_95850826�
0spectral_normalization_5/StatefulPartitionedCallStatefulPartitionedCall9spectral_normalization_4/StatefulPartitionedCall:output:0!spectral_normalization_5_95850844!spectral_normalization_5_95850846*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������2 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_95850843�
0spectral_normalization_6/StatefulPartitionedCallStatefulPartitionedCall9spectral_normalization_5/StatefulPartitionedCall:output:0!spectral_normalization_6_95850861!spectral_normalization_6_95850863*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *_
fZRX
V__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_95850860�
flatten/PartitionedCallPartitionedCall9spectral_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_95850872�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_95850885dense_95850887*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_95850884u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall1^spectral_normalization_4/StatefulPartitionedCall1^spectral_normalization_5/StatefulPartitionedCall1^spectral_normalization_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������2: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2d
0spectral_normalization_4/StatefulPartitionedCall0spectral_normalization_4/StatefulPartitionedCall2d
0spectral_normalization_5/StatefulPartitionedCall0spectral_normalization_5/StatefulPartitionedCall2d
0spectral_normalization_6/StatefulPartitionedCall0spectral_normalization_6/StatefulPartitionedCall:W S
/
_output_shapes
:���������2
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_28
serving_default_input_2:0���������29
dense0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	layer
w
w_shape
sn_u
u"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	layer
 w
!w_shape
"sn_u
"u"
_tf_keras_layer
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
	)layer
*w
+w_shape
,sn_u
,u"
_tf_keras_layer
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

9kernel
:bias"
_tf_keras_layer
n
0
;1
2
 3
<4
"5
*6
=7
,8
99
:10"
trackable_list_wrapper
X
0
;1
 2
<3
*4
=5
96
:7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ctrace_0
Dtrace_1
Etrace_2
Ftrace_32�
9__inference_discriminator_ensemble_layer_call_fn_95850910
9__inference_discriminator_ensemble_layer_call_fn_95851315
9__inference_discriminator_ensemble_layer_call_fn_95851342
9__inference_discriminator_ensemble_layer_call_fn_95851215�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zCtrace_0zDtrace_1zEtrace_2zFtrace_3
�
Gtrace_0
Htrace_1
Itrace_2
Jtrace_32�
T__inference_discriminator_ensemble_layer_call_and_return_conditional_losses_95851375
T__inference_discriminator_ensemble_layer_call_and_return_conditional_losses_95851495
T__inference_discriminator_ensemble_layer_call_and_return_conditional_losses_95851240
T__inference_discriminator_ensemble_layer_call_and_return_conditional_losses_95851271�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zGtrace_0zHtrace_1zItrace_2zJtrace_3
�B�
#__inference__wrapped_model_95850808input_2"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
Kserving_default"
signature_map
5
0
;1
2"
trackable_list_wrapper
.
0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Qtrace_0
Rtrace_12�
;__inference_spectral_normalization_4_layer_call_fn_95851504
;__inference_spectral_normalization_4_layer_call_fn_95851515�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zQtrace_0zRtrace_1
�
Strace_0
Ttrace_12�
V__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_95851526
V__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_95851566�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zStrace_0zTtrace_1
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses
[
activation

kernel
;bias
 \_jit_compiled_convolution_op"
_tf_keras_layer
9:7@2spectral_normalization_4/kernel
 "
trackable_list_wrapper
-:+@2spectral_normalization_4/sn_u
5
 0
<1
"2"
trackable_list_wrapper
.
 0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
btrace_0
ctrace_12�
;__inference_spectral_normalization_5_layer_call_fn_95851575
;__inference_spectral_normalization_5_layer_call_fn_95851586�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zbtrace_0zctrace_1
�
dtrace_0
etrace_12�
V__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_95851597
V__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_95851637�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zdtrace_0zetrace_1
�
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses
l
activation

 kernel
<bias
 m_jit_compiled_convolution_op"
_tf_keras_layer
9:7	@ 2spectral_normalization_5/kernel
 "
trackable_list_wrapper
-:+ 2spectral_normalization_5/sn_u
5
*0
=1
,2"
trackable_list_wrapper
.
*0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
�
strace_0
ttrace_12�
;__inference_spectral_normalization_6_layer_call_fn_95851646
;__inference_spectral_normalization_6_layer_call_fn_95851657�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zstrace_0zttrace_1
�
utrace_0
vtrace_12�
V__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_95851668
V__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_95851708�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zutrace_0zvtrace_1
�
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses
}
activation

*kernel
=bias
 ~_jit_compiled_convolution_op"
_tf_keras_layer
9:7 2spectral_normalization_6/kernel
 "
trackable_list_wrapper
-:+2spectral_normalization_6/sn_u
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_flatten_layer_call_fn_95851713�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_flatten_layer_call_and_return_conditional_losses_95851719�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_layer_call_fn_95851728�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_layer_call_and_return_conditional_losses_95851738�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
:	�2dense/kernel
:2
dense/bias
+:)@2spectral_normalization_4/bias
+:) 2spectral_normalization_5/bias
+:)2spectral_normalization_6/bias
5
0
"1
,2"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
9__inference_discriminator_ensemble_layer_call_fn_95850910input_2"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
9__inference_discriminator_ensemble_layer_call_fn_95851315inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
9__inference_discriminator_ensemble_layer_call_fn_95851342inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
9__inference_discriminator_ensemble_layer_call_fn_95851215input_2"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
T__inference_discriminator_ensemble_layer_call_and_return_conditional_losses_95851375inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
T__inference_discriminator_ensemble_layer_call_and_return_conditional_losses_95851495inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
T__inference_discriminator_ensemble_layer_call_and_return_conditional_losses_95851240input_2"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
T__inference_discriminator_ensemble_layer_call_and_return_conditional_losses_95851271input_2"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
&__inference_signature_wrapper_95851294input_2"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
;__inference_spectral_normalization_4_layer_call_fn_95851504inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
;__inference_spectral_normalization_4_layer_call_fn_95851515inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
V__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_95851526inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
V__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_95851566inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
.
0
;1"
trackable_list_wrapper
.
0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
'
"0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
;__inference_spectral_normalization_5_layer_call_fn_95851575inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
;__inference_spectral_normalization_5_layer_call_fn_95851586inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
V__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_95851597inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
V__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_95851637inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
.
 0
<1"
trackable_list_wrapper
.
 0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
'
,0"
trackable_list_wrapper
'
)0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
;__inference_spectral_normalization_6_layer_call_fn_95851646inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
;__inference_spectral_normalization_6_layer_call_fn_95851657inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
V__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_95851668inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
V__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_95851708inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
.
*0
=1"
trackable_list_wrapper
.
*0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_flatten_layer_call_fn_95851713inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_flatten_layer_call_and_return_conditional_losses_95851719inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_layer_call_fn_95851728inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_layer_call_and_return_conditional_losses_95851738inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
'
[0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
'
l0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
'
}0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper�
#__inference__wrapped_model_95850808s; <*=9:8�5
.�+
)�&
input_2���������2
� "-�*
(
dense�
dense����������
C__inference_dense_layer_call_and_return_conditional_losses_95851738]9:0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� |
(__inference_dense_layer_call_fn_95851728P9:0�-
&�#
!�
inputs����������
� "�����������
T__inference_discriminator_ensemble_layer_call_and_return_conditional_losses_95851240s; <*=9:@�=
6�3
)�&
input_2���������2
p 

 
� "%�"
�
0���������
� �
T__inference_discriminator_ensemble_layer_call_and_return_conditional_losses_95851271v; "<*,=9:@�=
6�3
)�&
input_2���������2
p

 
� "%�"
�
0���������
� �
T__inference_discriminator_ensemble_layer_call_and_return_conditional_losses_95851375r; <*=9:?�<
5�2
(�%
inputs���������2
p 

 
� "%�"
�
0���������
� �
T__inference_discriminator_ensemble_layer_call_and_return_conditional_losses_95851495u; "<*,=9:?�<
5�2
(�%
inputs���������2
p

 
� "%�"
�
0���������
� �
9__inference_discriminator_ensemble_layer_call_fn_95850910f; <*=9:@�=
6�3
)�&
input_2���������2
p 

 
� "�����������
9__inference_discriminator_ensemble_layer_call_fn_95851215i; "<*,=9:@�=
6�3
)�&
input_2���������2
p

 
� "�����������
9__inference_discriminator_ensemble_layer_call_fn_95851315e; <*=9:?�<
5�2
(�%
inputs���������2
p 

 
� "�����������
9__inference_discriminator_ensemble_layer_call_fn_95851342h; "<*,=9:?�<
5�2
(�%
inputs���������2
p

 
� "�����������
E__inference_flatten_layer_call_and_return_conditional_losses_95851719a7�4
-�*
(�%
inputs���������2
� "&�#
�
0����������
� �
*__inference_flatten_layer_call_fn_95851713T7�4
-�*
(�%
inputs���������2
� "������������
&__inference_signature_wrapper_95851294~; <*=9:C�@
� 
9�6
4
input_2)�&
input_2���������2"-�*
(
dense�
dense����������
V__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_95851526p;;�8
1�.
(�%
inputs���������2
p 
� "-�*
#� 
0���������2@
� �
V__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_95851566q;;�8
1�.
(�%
inputs���������2
p
� "-�*
#� 
0���������2@
� �
;__inference_spectral_normalization_4_layer_call_fn_95851504c;;�8
1�.
(�%
inputs���������2
p 
� " ����������2@�
;__inference_spectral_normalization_4_layer_call_fn_95851515d;;�8
1�.
(�%
inputs���������2
p
� " ����������2@�
V__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_95851597p <;�8
1�.
(�%
inputs���������2@
p 
� "-�*
#� 
0���������2 
� �
V__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_95851637q "<;�8
1�.
(�%
inputs���������2@
p
� "-�*
#� 
0���������2 
� �
;__inference_spectral_normalization_5_layer_call_fn_95851575c <;�8
1�.
(�%
inputs���������2@
p 
� " ����������2 �
;__inference_spectral_normalization_5_layer_call_fn_95851586d "<;�8
1�.
(�%
inputs���������2@
p
� " ����������2 �
V__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_95851668p*=;�8
1�.
(�%
inputs���������2 
p 
� "-�*
#� 
0���������2
� �
V__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_95851708q*,=;�8
1�.
(�%
inputs���������2 
p
� "-�*
#� 
0���������2
� �
;__inference_spectral_normalization_6_layer_call_fn_95851646c*=;�8
1�.
(�%
inputs���������2 
p 
� " ����������2�
;__inference_spectral_normalization_6_layer_call_fn_95851657d*,=;�8
1�.
(�%
inputs���������2 
p
� " ����������2
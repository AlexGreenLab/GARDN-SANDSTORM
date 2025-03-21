��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
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
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
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
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
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
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
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
 �"serve*2.10.02unknown8��
�
spectral_normalization_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name spectral_normalization_23/bias
�
2spectral_normalization_23/bias/Read/ReadVariableOpReadVariableOpspectral_normalization_23/bias*
_output_shapes
:*
dtype0
�
spectral_normalization_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name spectral_normalization_22/bias
�
2spectral_normalization_22/bias/Read/ReadVariableOpReadVariableOpspectral_normalization_22/bias*
_output_shapes
:*
dtype0
�
spectral_normalization_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name spectral_normalization_21/bias
�
2spectral_normalization_21/bias/Read/ReadVariableOpReadVariableOpspectral_normalization_21/bias*
_output_shapes
: *
dtype0
�
spectral_normalization_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name spectral_normalization_20/bias
�
2spectral_normalization_20/bias/Read/ReadVariableOpReadVariableOpspectral_normalization_20/bias*
_output_shapes
:@*
dtype0
�
spectral_normalization_23/sn_uVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name spectral_normalization_23/sn_u
�
2spectral_normalization_23/sn_u/Read/ReadVariableOpReadVariableOpspectral_normalization_23/sn_u*
_output_shapes

:*
dtype0
�
 spectral_normalization_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*1
shared_name" spectral_normalization_23/kernel
�
4spectral_normalization_23/kernel/Read/ReadVariableOpReadVariableOp spectral_normalization_23/kernel*
_output_shapes
:	�*
dtype0
�
layer_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_8/beta
�
.layer_normalization_8/beta/Read/ReadVariableOpReadVariableOplayer_normalization_8/beta*
_output_shapes
:*
dtype0
�
layer_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_8/gamma
�
/layer_normalization_8/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_8/gamma*
_output_shapes
:*
dtype0
�
spectral_normalization_22/sn_uVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name spectral_normalization_22/sn_u
�
2spectral_normalization_22/sn_u/Read/ReadVariableOpReadVariableOpspectral_normalization_22/sn_u*
_output_shapes

:*
dtype0
�
 spectral_normalization_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" spectral_normalization_22/kernel
�
4spectral_normalization_22/kernel/Read/ReadVariableOpReadVariableOp spectral_normalization_22/kernel*&
_output_shapes
: *
dtype0
�
layer_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namelayer_normalization_7/beta
�
.layer_normalization_7/beta/Read/ReadVariableOpReadVariableOplayer_normalization_7/beta*
_output_shapes
: *
dtype0
�
layer_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namelayer_normalization_7/gamma
�
/layer_normalization_7/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_7/gamma*
_output_shapes
: *
dtype0
�
spectral_normalization_21/sn_uVarHandleOp*
_output_shapes
: *
dtype0*
shape
: */
shared_name spectral_normalization_21/sn_u
�
2spectral_normalization_21/sn_u/Read/ReadVariableOpReadVariableOpspectral_normalization_21/sn_u*
_output_shapes

: *
dtype0
�
 spectral_normalization_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@ *1
shared_name" spectral_normalization_21/kernel
�
4spectral_normalization_21/kernel/Read/ReadVariableOpReadVariableOp spectral_normalization_21/kernel*&
_output_shapes
:	@ *
dtype0
�
layer_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namelayer_normalization_6/beta
�
.layer_normalization_6/beta/Read/ReadVariableOpReadVariableOplayer_normalization_6/beta*
_output_shapes
:@*
dtype0
�
layer_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namelayer_normalization_6/gamma
�
/layer_normalization_6/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_6/gamma*
_output_shapes
:@*
dtype0
�
spectral_normalization_20/sn_uVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*/
shared_name spectral_normalization_20/sn_u
�
2spectral_normalization_20/sn_u/Read/ReadVariableOpReadVariableOpspectral_normalization_20/sn_u*
_output_shapes

:@*
dtype0
�
 spectral_normalization_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" spectral_normalization_20/kernel
�
4spectral_normalization_20/kernel/Read/ReadVariableOpReadVariableOp spectral_normalization_20/kernel*&
_output_shapes
:@*
dtype0
�
serving_default_input_6Placeholder*/
_output_shapes
:���������<*
dtype0*$
shape:���������<
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_6 spectral_normalization_20/kernelspectral_normalization_20/biaslayer_normalization_6/gammalayer_normalization_6/beta spectral_normalization_21/kernelspectral_normalization_21/biaslayer_normalization_7/gammalayer_normalization_7/beta spectral_normalization_22/kernelspectral_normalization_22/biaslayer_normalization_8/gammalayer_normalization_8/beta spectral_normalization_23/kernelspectral_normalization_23/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_733759

NoOpNoOp
�U
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�U
value�TB�T B�T
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer-7
	layer_with_weights-6
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	layer
w
w_shape
sn_u
u*
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"axis
	#gamma
$beta*
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
	+layer
,w
-w_shape
.sn_u
.u*
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
5axis
	6gamma
7beta*
�
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses
	>layer
?w
@w_shape
Asn_u
Au*
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
Haxis
	Igamma
Jbeta*
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses* 
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses
	Wlayer
Xw
Yw_shape
Zsn_u
Zu*
�
0
[1
2
#3
$4
,5
\6
.7
68
79
?10
]11
A12
I13
J14
X15
^16
Z17*
j
0
[1
#2
$3
,4
\5
66
77
?8
]9
I10
J11
X12
^13*
* 
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
dtrace_0
etrace_1
ftrace_2
gtrace_3* 
6
htrace_0
itrace_1
jtrace_2
ktrace_3* 
* 

lserving_default* 

0
[1
2*

0
[1*
* 
�
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

rtrace_0
strace_1* 

ttrace_0
utrace_1* 
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses
|
activation

kernel
[bias
 }_jit_compiled_convolution_op*
ke
VARIABLE_VALUE spectral_normalization_20/kernel1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUE*
* 
lf
VARIABLE_VALUEspectral_normalization_20/sn_u4layer_with_weights-0/sn_u/.ATTRIBUTES/VARIABLE_VALUE*

#0
$1*

#0
$1*
* 
�
~non_trainable_variables

layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
jd
VARIABLE_VALUElayer_normalization_6/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_6/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*

,0
\1
.2*

,0
\1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
|
activation

,kernel
\bias
!�_jit_compiled_convolution_op*
ke
VARIABLE_VALUE spectral_normalization_21/kernel1layer_with_weights-2/w/.ATTRIBUTES/VARIABLE_VALUE*
* 
lf
VARIABLE_VALUEspectral_normalization_21/sn_u4layer_with_weights-2/sn_u/.ATTRIBUTES/VARIABLE_VALUE*

60
71*

60
71*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
jd
VARIABLE_VALUElayer_normalization_7/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_7/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*

?0
]1
A2*

?0
]1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
|
activation

?kernel
]bias
!�_jit_compiled_convolution_op*
ke
VARIABLE_VALUE spectral_normalization_22/kernel1layer_with_weights-4/w/.ATTRIBUTES/VARIABLE_VALUE*
* 
lf
VARIABLE_VALUEspectral_normalization_22/sn_u4layer_with_weights-4/sn_u/.ATTRIBUTES/VARIABLE_VALUE*

I0
J1*

I0
J1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
jd
VARIABLE_VALUElayer_normalization_8/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_8/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

X0
^1
Z2*

X0
^1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Xkernel
^bias*
ke
VARIABLE_VALUE spectral_normalization_23/kernel1layer_with_weights-6/w/.ATTRIBUTES/VARIABLE_VALUE*
* 
lf
VARIABLE_VALUEspectral_normalization_23/sn_u4layer_with_weights-6/sn_u/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEspectral_normalization_20/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEspectral_normalization_21/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEspectral_normalization_22/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEspectral_normalization_23/bias'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
 
0
.1
A2
Z3*
C
0
1
2
3
4
5
6
7
	8*
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

0*

0*
* 
* 
* 
* 
* 
* 
* 

0
[1*

0
[1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses*
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

.0*

+0*
* 
* 
* 
* 
* 
* 
* 

,0
\1*

,0
\1*
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
'�"call_and_return_conditional_losses*
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

A0*

>0*
* 
* 
* 
* 
* 
* 
* 

?0
]1*

?0
]1*
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
'�"call_and_return_conditional_losses*
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

Z0*

W0*
* 
* 
* 
* 
* 
* 
* 

X0
^1*

X0
^1*
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
'�"call_and_return_conditional_losses*
* 
* 
* 
	
|0* 
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
	
|0* 
* 
* 
* 
* 
	
|0* 
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
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename4spectral_normalization_20/kernel/Read/ReadVariableOp2spectral_normalization_20/sn_u/Read/ReadVariableOp/layer_normalization_6/gamma/Read/ReadVariableOp.layer_normalization_6/beta/Read/ReadVariableOp4spectral_normalization_21/kernel/Read/ReadVariableOp2spectral_normalization_21/sn_u/Read/ReadVariableOp/layer_normalization_7/gamma/Read/ReadVariableOp.layer_normalization_7/beta/Read/ReadVariableOp4spectral_normalization_22/kernel/Read/ReadVariableOp2spectral_normalization_22/sn_u/Read/ReadVariableOp/layer_normalization_8/gamma/Read/ReadVariableOp.layer_normalization_8/beta/Read/ReadVariableOp4spectral_normalization_23/kernel/Read/ReadVariableOp2spectral_normalization_23/sn_u/Read/ReadVariableOp2spectral_normalization_20/bias/Read/ReadVariableOp2spectral_normalization_21/bias/Read/ReadVariableOp2spectral_normalization_22/bias/Read/ReadVariableOp2spectral_normalization_23/bias/Read/ReadVariableOpConst*
Tin
2*
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
GPU2*0J 8� *(
f#R!
__inference__traced_save_734856
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename spectral_normalization_20/kernelspectral_normalization_20/sn_ulayer_normalization_6/gammalayer_normalization_6/beta spectral_normalization_21/kernelspectral_normalization_21/sn_ulayer_normalization_7/gammalayer_normalization_7/beta spectral_normalization_22/kernelspectral_normalization_22/sn_ulayer_normalization_8/gammalayer_normalization_8/beta spectral_normalization_23/kernelspectral_normalization_23/sn_uspectral_normalization_20/biasspectral_normalization_21/biasspectral_normalization_22/biasspectral_normalization_23/bias*
Tin
2*
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
GPU2*0J 8� *+
f&R$
"__inference__traced_restore_734920��
�
�
U__inference_spectral_normalization_23_layer_call_and_return_conditional_losses_734740

inputs9
&dense_2_matmul_readvariableop_resource:	�5
'dense_2_biasadd_readvariableop_resource:
identity��dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0y
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
F
*__inference_flatten_2_layer_call_fn_734704

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
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_733144a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
U__inference_spectral_normalization_22_layer_call_and_return_conditional_losses_734598

inputsB
(conv2d_20_conv2d_readvariableop_resource: 7
)conv2d_20_biasadd_readvariableop_resource:
identity�� conv2d_20/BiasAdd/ReadVariableOp�conv2d_20/Conv2D/ReadVariableOp�
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_20/Conv2DConv2Dinputs'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<*
paddingSAME*
strides
�
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<{
!conv2d_20/leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_20/BiasAdd:output:0*/
_output_shapes
:���������<�
IdentityIdentity/conv2d_20/leaky_re_lu_2/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������<�
NoOpNoOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������< : : 2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������< 
 
_user_specified_nameinputs
�
�
6__inference_layer_normalization_8_layer_call_fn_734647

inputs
unknown:
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
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_layer_normalization_8_layer_call_and_return_conditional_losses_733132w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�7
�
U__inference_spectral_normalization_22_layer_call_and_return_conditional_losses_733322

inputs9
reshape_readvariableop_resource: C
1spectral_normalize_matmul_readvariableop_resource:7
)conv2d_20_biasadd_readvariableop_resource:
identity��Reshape/ReadVariableOp� conv2d_20/BiasAdd/ReadVariableOp�conv2d_20/Conv2D/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOp~
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
conv2d_20/Conv2D/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
: *
dtype0�
conv2d_20/Conv2DConv2Dinputs'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<*
paddingSAME*
strides
�
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<{
!conv2d_20/leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_20/BiasAdd:output:0*/
_output_shapes
:���������<�
IdentityIdentity/conv2d_20/leaky_re_lu_2/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������<�
NoOpNoOp^Reshape/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������< : : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:���������< 
 
_user_specified_nameinputs
�)
�
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_734435

inputs+
mul_4_readvariableop_resource:@)
add_readvariableop_resource:@
identity��add/ReadVariableOp�mul_4/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
mul_2Mul	mul_1:z:0strided_slice_2:output:0*
T0*
_output_shapes
: _
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_3Mulmul_3/x:output:0strided_slice_3:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackReshape/shape/0:output:0	mul_2:z:0	mul_3:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������@L
ones/packedPack	mul_2:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:���������M
zeros/packedPack	mul_2:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:���������H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB �
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������@:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:t
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*/
_output_shapes
:���������<@n
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes
:@*
dtype0x
mul_4MulReshape_1:output:0mul_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<@j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
addAddV2	mul_4:z:0add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<@^
IdentityIdentityadd:z:0^NoOp*
T0*/
_output_shapes
:���������<@r
NoOpNoOp^add/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<@: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:W S
/
_output_shapes
:���������<@
 
_user_specified_nameinputs
�
�
:__inference_spectral_normalization_23_layer_call_fn_734730

inputs
unknown:	�
	unknown_0:
	unknown_1:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_23_layer_call_and_return_conditional_losses_733245o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:����������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
:__inference_spectral_normalization_20_layer_call_fn_734312

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
:���������<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_20_layer_call_and_return_conditional_losses_732924w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�)
�
Q__inference_layer_normalization_8_layer_call_and_return_conditional_losses_733132

inputs+
mul_4_readvariableop_resource:)
add_readvariableop_resource:
identity��add/ReadVariableOp�mul_4/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
mul_2Mul	mul_1:z:0strided_slice_2:output:0*
T0*
_output_shapes
: _
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_3Mulmul_3/x:output:0strided_slice_3:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackReshape/shape/0:output:0	mul_2:z:0	mul_3:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������L
ones/packedPack	mul_2:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:���������M
zeros/packedPack	mul_2:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:���������H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB �
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:t
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*/
_output_shapes
:���������<n
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes
:*
dtype0x
mul_4MulReshape_1:output:0mul_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0m
addAddV2	mul_4:z:0add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<^
IdentityIdentityadd:z:0^NoOp*
T0*/
_output_shapes
:���������<r
NoOpNoOp^add/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
U__inference_spectral_normalization_22_layer_call_and_return_conditional_losses_733074

inputsB
(conv2d_20_conv2d_readvariableop_resource: 7
)conv2d_20_biasadd_readvariableop_resource:
identity�� conv2d_20/BiasAdd/ReadVariableOp�conv2d_20/Conv2D/ReadVariableOp�
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_20/Conv2DConv2Dinputs'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<*
paddingSAME*
strides
�
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<{
!conv2d_20/leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_20/BiasAdd:output:0*/
_output_shapes
:���������<�
IdentityIdentity/conv2d_20/leaky_re_lu_2/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������<�
NoOpNoOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������< : : 2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������< 
 
_user_specified_nameinputs
�
�
:__inference_spectral_normalization_21_layer_call_fn_734444

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
:���������< *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_21_layer_call_and_return_conditional_losses_732999w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������< `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������<@
 
_user_specified_nameinputs
�7
�
U__inference_spectral_normalization_20_layer_call_and_return_conditional_losses_734374

inputs9
reshape_readvariableop_resource:@C
1spectral_normalize_matmul_readvariableop_resource:@7
)conv2d_18_biasadd_readvariableop_resource:@
identity��Reshape/ReadVariableOp� conv2d_18/BiasAdd/ReadVariableOp�conv2d_18/Conv2D/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOp~
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
conv2d_18/Conv2D/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
:@*
dtype0�
conv2d_18/Conv2DConv2Dinputs'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<@*
paddingSAME*
strides
�
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<@{
!conv2d_18/leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_18/BiasAdd:output:0*/
_output_shapes
:���������<@�
IdentityIdentity/conv2d_18/leaky_re_lu_2/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������<@�
NoOpNoOp^Reshape/ReadVariableOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������<: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�4
�

I__inference_discriminator_layer_call_and_return_conditional_losses_733556

inputs:
 spectral_normalization_20_733511:@2
 spectral_normalization_20_733513:@.
 spectral_normalization_20_733515:@*
layer_normalization_6_733518:@*
layer_normalization_6_733520:@:
 spectral_normalization_21_733523:	@ 2
 spectral_normalization_21_733525: .
 spectral_normalization_21_733527: *
layer_normalization_7_733530: *
layer_normalization_7_733532: :
 spectral_normalization_22_733535: 2
 spectral_normalization_22_733537:.
 spectral_normalization_22_733539:*
layer_normalization_8_733542:*
layer_normalization_8_733544:3
 spectral_normalization_23_733548:	�2
 spectral_normalization_23_733550:.
 spectral_normalization_23_733552:
identity��-layer_normalization_6/StatefulPartitionedCall�-layer_normalization_7/StatefulPartitionedCall�-layer_normalization_8/StatefulPartitionedCall�1spectral_normalization_20/StatefulPartitionedCall�1spectral_normalization_21/StatefulPartitionedCall�1spectral_normalization_22/StatefulPartitionedCall�1spectral_normalization_23/StatefulPartitionedCall�
1spectral_normalization_20/StatefulPartitionedCallStatefulPartitionedCallinputs spectral_normalization_20_733511 spectral_normalization_20_733513 spectral_normalization_20_733515*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_20_layer_call_and_return_conditional_losses_733464�
-layer_normalization_6/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_20/StatefulPartitionedCall:output:0layer_normalization_6_733518layer_normalization_6_733520*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_732982�
1spectral_normalization_21/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_6/StatefulPartitionedCall:output:0 spectral_normalization_21_733523 spectral_normalization_21_733525 spectral_normalization_21_733527*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������< *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_21_layer_call_and_return_conditional_losses_733393�
-layer_normalization_7/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_21/StatefulPartitionedCall:output:0layer_normalization_7_733530layer_normalization_7_733532*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������< *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_733057�
1spectral_normalization_22/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_7/StatefulPartitionedCall:output:0 spectral_normalization_22_733535 spectral_normalization_22_733537 spectral_normalization_22_733539*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_22_layer_call_and_return_conditional_losses_733322�
-layer_normalization_8/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_22/StatefulPartitionedCall:output:0layer_normalization_8_733542layer_normalization_8_733544*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_layer_normalization_8_layer_call_and_return_conditional_losses_733132�
flatten_2/PartitionedCallPartitionedCall6layer_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_733144�
1spectral_normalization_23/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0 spectral_normalization_23_733548 spectral_normalization_23_733550 spectral_normalization_23_733552*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_23_layer_call_and_return_conditional_losses_733245�
IdentityIdentity:spectral_normalization_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^layer_normalization_6/StatefulPartitionedCall.^layer_normalization_7/StatefulPartitionedCall.^layer_normalization_8/StatefulPartitionedCall2^spectral_normalization_20/StatefulPartitionedCall2^spectral_normalization_21/StatefulPartitionedCall2^spectral_normalization_22/StatefulPartitionedCall2^spectral_normalization_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������<: : : : : : : : : : : : : : : : : : 2^
-layer_normalization_6/StatefulPartitionedCall-layer_normalization_6/StatefulPartitionedCall2^
-layer_normalization_7/StatefulPartitionedCall-layer_normalization_7/StatefulPartitionedCall2^
-layer_normalization_8/StatefulPartitionedCall-layer_normalization_8/StatefulPartitionedCall2f
1spectral_normalization_20/StatefulPartitionedCall1spectral_normalization_20/StatefulPartitionedCall2f
1spectral_normalization_21/StatefulPartitionedCall1spectral_normalization_21/StatefulPartitionedCall2f
1spectral_normalization_22/StatefulPartitionedCall1spectral_normalization_22/StatefulPartitionedCall2f
1spectral_normalization_23/StatefulPartitionedCall1spectral_normalization_23/StatefulPartitionedCall:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
��
�
I__inference_discriminator_layer_call_and_return_conditional_losses_734010

inputs\
Bspectral_normalization_20_conv2d_18_conv2d_readvariableop_resource:@Q
Cspectral_normalization_20_conv2d_18_biasadd_readvariableop_resource:@A
3layer_normalization_6_mul_4_readvariableop_resource:@?
1layer_normalization_6_add_readvariableop_resource:@\
Bspectral_normalization_21_conv2d_19_conv2d_readvariableop_resource:	@ Q
Cspectral_normalization_21_conv2d_19_biasadd_readvariableop_resource: A
3layer_normalization_7_mul_4_readvariableop_resource: ?
1layer_normalization_7_add_readvariableop_resource: \
Bspectral_normalization_22_conv2d_20_conv2d_readvariableop_resource: Q
Cspectral_normalization_22_conv2d_20_biasadd_readvariableop_resource:A
3layer_normalization_8_mul_4_readvariableop_resource:?
1layer_normalization_8_add_readvariableop_resource:S
@spectral_normalization_23_dense_2_matmul_readvariableop_resource:	�O
Aspectral_normalization_23_dense_2_biasadd_readvariableop_resource:
identity��(layer_normalization_6/add/ReadVariableOp�*layer_normalization_6/mul_4/ReadVariableOp�(layer_normalization_7/add/ReadVariableOp�*layer_normalization_7/mul_4/ReadVariableOp�(layer_normalization_8/add/ReadVariableOp�*layer_normalization_8/mul_4/ReadVariableOp�:spectral_normalization_20/conv2d_18/BiasAdd/ReadVariableOp�9spectral_normalization_20/conv2d_18/Conv2D/ReadVariableOp�:spectral_normalization_21/conv2d_19/BiasAdd/ReadVariableOp�9spectral_normalization_21/conv2d_19/Conv2D/ReadVariableOp�:spectral_normalization_22/conv2d_20/BiasAdd/ReadVariableOp�9spectral_normalization_22/conv2d_20/Conv2D/ReadVariableOp�8spectral_normalization_23/dense_2/BiasAdd/ReadVariableOp�7spectral_normalization_23/dense_2/MatMul/ReadVariableOp�
9spectral_normalization_20/conv2d_18/Conv2D/ReadVariableOpReadVariableOpBspectral_normalization_20_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
*spectral_normalization_20/conv2d_18/Conv2DConv2DinputsAspectral_normalization_20/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<@*
paddingSAME*
strides
�
:spectral_normalization_20/conv2d_18/BiasAdd/ReadVariableOpReadVariableOpCspectral_normalization_20_conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
+spectral_normalization_20/conv2d_18/BiasAddBiasAdd3spectral_normalization_20/conv2d_18/Conv2D:output:0Bspectral_normalization_20/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<@�
;spectral_normalization_20/conv2d_18/leaky_re_lu_2/LeakyRelu	LeakyRelu4spectral_normalization_20/conv2d_18/BiasAdd:output:0*/
_output_shapes
:���������<@�
layer_normalization_6/ShapeShapeIspectral_normalization_20/conv2d_18/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:s
)layer_normalization_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_6/strided_sliceStridedSlice$layer_normalization_6/Shape:output:02layer_normalization_6/strided_slice/stack:output:04layer_normalization_6/strided_slice/stack_1:output:04layer_normalization_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_6/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_6/mulMul$layer_normalization_6/mul/x:output:0,layer_normalization_6/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_6/strided_slice_1StridedSlice$layer_normalization_6/Shape:output:04layer_normalization_6/strided_slice_1/stack:output:06layer_normalization_6/strided_slice_1/stack_1:output:06layer_normalization_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_6/mul_1Mullayer_normalization_6/mul:z:0.layer_normalization_6/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_6/strided_slice_2StridedSlice$layer_normalization_6/Shape:output:04layer_normalization_6/strided_slice_2/stack:output:06layer_normalization_6/strided_slice_2/stack_1:output:06layer_normalization_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_6/mul_2Mullayer_normalization_6/mul_1:z:0.layer_normalization_6/strided_slice_2:output:0*
T0*
_output_shapes
: u
+layer_normalization_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_6/strided_slice_3StridedSlice$layer_normalization_6/Shape:output:04layer_normalization_6/strided_slice_3/stack:output:06layer_normalization_6/strided_slice_3/stack_1:output:06layer_normalization_6/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_6/mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_6/mul_3Mul&layer_normalization_6/mul_3/x:output:0.layer_normalization_6/strided_slice_3:output:0*
T0*
_output_shapes
: g
%layer_normalization_6/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_6/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
#layer_normalization_6/Reshape/shapePack.layer_normalization_6/Reshape/shape/0:output:0layer_normalization_6/mul_2:z:0layer_normalization_6/mul_3:z:0.layer_normalization_6/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
layer_normalization_6/ReshapeReshapeIspectral_normalization_20/conv2d_18/leaky_re_lu_2/LeakyRelu:activations:0,layer_normalization_6/Reshape/shape:output:0*
T0*/
_output_shapes
:���������@x
!layer_normalization_6/ones/packedPacklayer_normalization_6/mul_2:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_6/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
layer_normalization_6/onesFill*layer_normalization_6/ones/packed:output:0)layer_normalization_6/ones/Const:output:0*
T0*#
_output_shapes
:���������y
"layer_normalization_6/zeros/packedPacklayer_normalization_6/mul_2:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
layer_normalization_6/zerosFill+layer_normalization_6/zeros/packed:output:0*layer_normalization_6/zeros/Const:output:0*
T0*#
_output_shapes
:���������^
layer_normalization_6/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
&layer_normalization_6/FusedBatchNormV3FusedBatchNormV3&layer_normalization_6/Reshape:output:0#layer_normalization_6/ones:output:0$layer_normalization_6/zeros:output:0$layer_normalization_6/Const:output:0&layer_normalization_6/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������@:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
layer_normalization_6/Reshape_1Reshape*layer_normalization_6/FusedBatchNormV3:y:0$layer_normalization_6/Shape:output:0*
T0*/
_output_shapes
:���������<@�
*layer_normalization_6/mul_4/ReadVariableOpReadVariableOp3layer_normalization_6_mul_4_readvariableop_resource*
_output_shapes
:@*
dtype0�
layer_normalization_6/mul_4Mul(layer_normalization_6/Reshape_1:output:02layer_normalization_6/mul_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<@�
(layer_normalization_6/add/ReadVariableOpReadVariableOp1layer_normalization_6_add_readvariableop_resource*
_output_shapes
:@*
dtype0�
layer_normalization_6/addAddV2layer_normalization_6/mul_4:z:00layer_normalization_6/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<@�
9spectral_normalization_21/conv2d_19/Conv2D/ReadVariableOpReadVariableOpBspectral_normalization_21_conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:	@ *
dtype0�
*spectral_normalization_21/conv2d_19/Conv2DConv2Dlayer_normalization_6/add:z:0Aspectral_normalization_21/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< *
paddingSAME*
strides
�
:spectral_normalization_21/conv2d_19/BiasAdd/ReadVariableOpReadVariableOpCspectral_normalization_21_conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
+spectral_normalization_21/conv2d_19/BiasAddBiasAdd3spectral_normalization_21/conv2d_19/Conv2D:output:0Bspectral_normalization_21/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< �
;spectral_normalization_21/conv2d_19/leaky_re_lu_2/LeakyRelu	LeakyRelu4spectral_normalization_21/conv2d_19/BiasAdd:output:0*/
_output_shapes
:���������< �
layer_normalization_7/ShapeShapeIspectral_normalization_21/conv2d_19/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:s
)layer_normalization_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_7/strided_sliceStridedSlice$layer_normalization_7/Shape:output:02layer_normalization_7/strided_slice/stack:output:04layer_normalization_7/strided_slice/stack_1:output:04layer_normalization_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_7/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_7/mulMul$layer_normalization_7/mul/x:output:0,layer_normalization_7/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_7/strided_slice_1StridedSlice$layer_normalization_7/Shape:output:04layer_normalization_7/strided_slice_1/stack:output:06layer_normalization_7/strided_slice_1/stack_1:output:06layer_normalization_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_7/mul_1Mullayer_normalization_7/mul:z:0.layer_normalization_7/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_7/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_7/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_7/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_7/strided_slice_2StridedSlice$layer_normalization_7/Shape:output:04layer_normalization_7/strided_slice_2/stack:output:06layer_normalization_7/strided_slice_2/stack_1:output:06layer_normalization_7/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_7/mul_2Mullayer_normalization_7/mul_1:z:0.layer_normalization_7/strided_slice_2:output:0*
T0*
_output_shapes
: u
+layer_normalization_7/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_7/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_7/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_7/strided_slice_3StridedSlice$layer_normalization_7/Shape:output:04layer_normalization_7/strided_slice_3/stack:output:06layer_normalization_7/strided_slice_3/stack_1:output:06layer_normalization_7/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_7/mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_7/mul_3Mul&layer_normalization_7/mul_3/x:output:0.layer_normalization_7/strided_slice_3:output:0*
T0*
_output_shapes
: g
%layer_normalization_7/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_7/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
#layer_normalization_7/Reshape/shapePack.layer_normalization_7/Reshape/shape/0:output:0layer_normalization_7/mul_2:z:0layer_normalization_7/mul_3:z:0.layer_normalization_7/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
layer_normalization_7/ReshapeReshapeIspectral_normalization_21/conv2d_19/leaky_re_lu_2/LeakyRelu:activations:0,layer_normalization_7/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� x
!layer_normalization_7/ones/packedPacklayer_normalization_7/mul_2:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_7/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
layer_normalization_7/onesFill*layer_normalization_7/ones/packed:output:0)layer_normalization_7/ones/Const:output:0*
T0*#
_output_shapes
:���������y
"layer_normalization_7/zeros/packedPacklayer_normalization_7/mul_2:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
layer_normalization_7/zerosFill+layer_normalization_7/zeros/packed:output:0*layer_normalization_7/zeros/Const:output:0*
T0*#
_output_shapes
:���������^
layer_normalization_7/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
&layer_normalization_7/FusedBatchNormV3FusedBatchNormV3&layer_normalization_7/Reshape:output:0#layer_normalization_7/ones:output:0$layer_normalization_7/zeros:output:0$layer_normalization_7/Const:output:0&layer_normalization_7/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:��������� :���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
layer_normalization_7/Reshape_1Reshape*layer_normalization_7/FusedBatchNormV3:y:0$layer_normalization_7/Shape:output:0*
T0*/
_output_shapes
:���������< �
*layer_normalization_7/mul_4/ReadVariableOpReadVariableOp3layer_normalization_7_mul_4_readvariableop_resource*
_output_shapes
: *
dtype0�
layer_normalization_7/mul_4Mul(layer_normalization_7/Reshape_1:output:02layer_normalization_7/mul_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< �
(layer_normalization_7/add/ReadVariableOpReadVariableOp1layer_normalization_7_add_readvariableop_resource*
_output_shapes
: *
dtype0�
layer_normalization_7/addAddV2layer_normalization_7/mul_4:z:00layer_normalization_7/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< �
9spectral_normalization_22/conv2d_20/Conv2D/ReadVariableOpReadVariableOpBspectral_normalization_22_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
*spectral_normalization_22/conv2d_20/Conv2DConv2Dlayer_normalization_7/add:z:0Aspectral_normalization_22/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<*
paddingSAME*
strides
�
:spectral_normalization_22/conv2d_20/BiasAdd/ReadVariableOpReadVariableOpCspectral_normalization_22_conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
+spectral_normalization_22/conv2d_20/BiasAddBiasAdd3spectral_normalization_22/conv2d_20/Conv2D:output:0Bspectral_normalization_22/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<�
;spectral_normalization_22/conv2d_20/leaky_re_lu_2/LeakyRelu	LeakyRelu4spectral_normalization_22/conv2d_20/BiasAdd:output:0*/
_output_shapes
:���������<�
layer_normalization_8/ShapeShapeIspectral_normalization_22/conv2d_20/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:s
)layer_normalization_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_8/strided_sliceStridedSlice$layer_normalization_8/Shape:output:02layer_normalization_8/strided_slice/stack:output:04layer_normalization_8/strided_slice/stack_1:output:04layer_normalization_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_8/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_8/mulMul$layer_normalization_8/mul/x:output:0,layer_normalization_8/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_8/strided_slice_1StridedSlice$layer_normalization_8/Shape:output:04layer_normalization_8/strided_slice_1/stack:output:06layer_normalization_8/strided_slice_1/stack_1:output:06layer_normalization_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_8/mul_1Mullayer_normalization_8/mul:z:0.layer_normalization_8/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_8/strided_slice_2StridedSlice$layer_normalization_8/Shape:output:04layer_normalization_8/strided_slice_2/stack:output:06layer_normalization_8/strided_slice_2/stack_1:output:06layer_normalization_8/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_8/mul_2Mullayer_normalization_8/mul_1:z:0.layer_normalization_8/strided_slice_2:output:0*
T0*
_output_shapes
: u
+layer_normalization_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_8/strided_slice_3StridedSlice$layer_normalization_8/Shape:output:04layer_normalization_8/strided_slice_3/stack:output:06layer_normalization_8/strided_slice_3/stack_1:output:06layer_normalization_8/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_8/mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_8/mul_3Mul&layer_normalization_8/mul_3/x:output:0.layer_normalization_8/strided_slice_3:output:0*
T0*
_output_shapes
: g
%layer_normalization_8/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
#layer_normalization_8/Reshape/shapePack.layer_normalization_8/Reshape/shape/0:output:0layer_normalization_8/mul_2:z:0layer_normalization_8/mul_3:z:0.layer_normalization_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
layer_normalization_8/ReshapeReshapeIspectral_normalization_22/conv2d_20/leaky_re_lu_2/LeakyRelu:activations:0,layer_normalization_8/Reshape/shape:output:0*
T0*/
_output_shapes
:���������x
!layer_normalization_8/ones/packedPacklayer_normalization_8/mul_2:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_8/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
layer_normalization_8/onesFill*layer_normalization_8/ones/packed:output:0)layer_normalization_8/ones/Const:output:0*
T0*#
_output_shapes
:���������y
"layer_normalization_8/zeros/packedPacklayer_normalization_8/mul_2:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
layer_normalization_8/zerosFill+layer_normalization_8/zeros/packed:output:0*layer_normalization_8/zeros/Const:output:0*
T0*#
_output_shapes
:���������^
layer_normalization_8/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
&layer_normalization_8/FusedBatchNormV3FusedBatchNormV3&layer_normalization_8/Reshape:output:0#layer_normalization_8/ones:output:0$layer_normalization_8/zeros:output:0$layer_normalization_8/Const:output:0&layer_normalization_8/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
layer_normalization_8/Reshape_1Reshape*layer_normalization_8/FusedBatchNormV3:y:0$layer_normalization_8/Shape:output:0*
T0*/
_output_shapes
:���������<�
*layer_normalization_8/mul_4/ReadVariableOpReadVariableOp3layer_normalization_8_mul_4_readvariableop_resource*
_output_shapes
:*
dtype0�
layer_normalization_8/mul_4Mul(layer_normalization_8/Reshape_1:output:02layer_normalization_8/mul_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<�
(layer_normalization_8/add/ReadVariableOpReadVariableOp1layer_normalization_8_add_readvariableop_resource*
_output_shapes
:*
dtype0�
layer_normalization_8/addAddV2layer_normalization_8/mul_4:z:00layer_normalization_8/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  �
flatten_2/ReshapeReshapelayer_normalization_8/add:z:0flatten_2/Const:output:0*
T0*(
_output_shapes
:�����������
7spectral_normalization_23/dense_2/MatMul/ReadVariableOpReadVariableOp@spectral_normalization_23_dense_2_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
(spectral_normalization_23/dense_2/MatMulMatMulflatten_2/Reshape:output:0?spectral_normalization_23/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
8spectral_normalization_23/dense_2/BiasAdd/ReadVariableOpReadVariableOpAspectral_normalization_23_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
)spectral_normalization_23/dense_2/BiasAddBiasAdd2spectral_normalization_23/dense_2/MatMul:product:0@spectral_normalization_23/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
IdentityIdentity2spectral_normalization_23/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp)^layer_normalization_6/add/ReadVariableOp+^layer_normalization_6/mul_4/ReadVariableOp)^layer_normalization_7/add/ReadVariableOp+^layer_normalization_7/mul_4/ReadVariableOp)^layer_normalization_8/add/ReadVariableOp+^layer_normalization_8/mul_4/ReadVariableOp;^spectral_normalization_20/conv2d_18/BiasAdd/ReadVariableOp:^spectral_normalization_20/conv2d_18/Conv2D/ReadVariableOp;^spectral_normalization_21/conv2d_19/BiasAdd/ReadVariableOp:^spectral_normalization_21/conv2d_19/Conv2D/ReadVariableOp;^spectral_normalization_22/conv2d_20/BiasAdd/ReadVariableOp:^spectral_normalization_22/conv2d_20/Conv2D/ReadVariableOp9^spectral_normalization_23/dense_2/BiasAdd/ReadVariableOp8^spectral_normalization_23/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������<: : : : : : : : : : : : : : 2T
(layer_normalization_6/add/ReadVariableOp(layer_normalization_6/add/ReadVariableOp2X
*layer_normalization_6/mul_4/ReadVariableOp*layer_normalization_6/mul_4/ReadVariableOp2T
(layer_normalization_7/add/ReadVariableOp(layer_normalization_7/add/ReadVariableOp2X
*layer_normalization_7/mul_4/ReadVariableOp*layer_normalization_7/mul_4/ReadVariableOp2T
(layer_normalization_8/add/ReadVariableOp(layer_normalization_8/add/ReadVariableOp2X
*layer_normalization_8/mul_4/ReadVariableOp*layer_normalization_8/mul_4/ReadVariableOp2x
:spectral_normalization_20/conv2d_18/BiasAdd/ReadVariableOp:spectral_normalization_20/conv2d_18/BiasAdd/ReadVariableOp2v
9spectral_normalization_20/conv2d_18/Conv2D/ReadVariableOp9spectral_normalization_20/conv2d_18/Conv2D/ReadVariableOp2x
:spectral_normalization_21/conv2d_19/BiasAdd/ReadVariableOp:spectral_normalization_21/conv2d_19/BiasAdd/ReadVariableOp2v
9spectral_normalization_21/conv2d_19/Conv2D/ReadVariableOp9spectral_normalization_21/conv2d_19/Conv2D/ReadVariableOp2x
:spectral_normalization_22/conv2d_20/BiasAdd/ReadVariableOp:spectral_normalization_22/conv2d_20/BiasAdd/ReadVariableOp2v
9spectral_normalization_22/conv2d_20/Conv2D/ReadVariableOp9spectral_normalization_22/conv2d_20/Conv2D/ReadVariableOp2t
8spectral_normalization_23/dense_2/BiasAdd/ReadVariableOp8spectral_normalization_23/dense_2/BiasAdd/ReadVariableOp2r
7spectral_normalization_23/dense_2/MatMul/ReadVariableOp7spectral_normalization_23/dense_2/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�4
�

I__inference_discriminator_layer_call_and_return_conditional_losses_733724
input_6:
 spectral_normalization_20_733679:@2
 spectral_normalization_20_733681:@.
 spectral_normalization_20_733683:@*
layer_normalization_6_733686:@*
layer_normalization_6_733688:@:
 spectral_normalization_21_733691:	@ 2
 spectral_normalization_21_733693: .
 spectral_normalization_21_733695: *
layer_normalization_7_733698: *
layer_normalization_7_733700: :
 spectral_normalization_22_733703: 2
 spectral_normalization_22_733705:.
 spectral_normalization_22_733707:*
layer_normalization_8_733710:*
layer_normalization_8_733712:3
 spectral_normalization_23_733716:	�2
 spectral_normalization_23_733718:.
 spectral_normalization_23_733720:
identity��-layer_normalization_6/StatefulPartitionedCall�-layer_normalization_7/StatefulPartitionedCall�-layer_normalization_8/StatefulPartitionedCall�1spectral_normalization_20/StatefulPartitionedCall�1spectral_normalization_21/StatefulPartitionedCall�1spectral_normalization_22/StatefulPartitionedCall�1spectral_normalization_23/StatefulPartitionedCall�
1spectral_normalization_20/StatefulPartitionedCallStatefulPartitionedCallinput_6 spectral_normalization_20_733679 spectral_normalization_20_733681 spectral_normalization_20_733683*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_20_layer_call_and_return_conditional_losses_733464�
-layer_normalization_6/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_20/StatefulPartitionedCall:output:0layer_normalization_6_733686layer_normalization_6_733688*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_732982�
1spectral_normalization_21/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_6/StatefulPartitionedCall:output:0 spectral_normalization_21_733691 spectral_normalization_21_733693 spectral_normalization_21_733695*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������< *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_21_layer_call_and_return_conditional_losses_733393�
-layer_normalization_7/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_21/StatefulPartitionedCall:output:0layer_normalization_7_733698layer_normalization_7_733700*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������< *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_733057�
1spectral_normalization_22/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_7/StatefulPartitionedCall:output:0 spectral_normalization_22_733703 spectral_normalization_22_733705 spectral_normalization_22_733707*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_22_layer_call_and_return_conditional_losses_733322�
-layer_normalization_8/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_22/StatefulPartitionedCall:output:0layer_normalization_8_733710layer_normalization_8_733712*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_layer_normalization_8_layer_call_and_return_conditional_losses_733132�
flatten_2/PartitionedCallPartitionedCall6layer_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_733144�
1spectral_normalization_23/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0 spectral_normalization_23_733716 spectral_normalization_23_733718 spectral_normalization_23_733720*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_23_layer_call_and_return_conditional_losses_733245�
IdentityIdentity:spectral_normalization_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^layer_normalization_6/StatefulPartitionedCall.^layer_normalization_7/StatefulPartitionedCall.^layer_normalization_8/StatefulPartitionedCall2^spectral_normalization_20/StatefulPartitionedCall2^spectral_normalization_21/StatefulPartitionedCall2^spectral_normalization_22/StatefulPartitionedCall2^spectral_normalization_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������<: : : : : : : : : : : : : : : : : : 2^
-layer_normalization_6/StatefulPartitionedCall-layer_normalization_6/StatefulPartitionedCall2^
-layer_normalization_7/StatefulPartitionedCall-layer_normalization_7/StatefulPartitionedCall2^
-layer_normalization_8/StatefulPartitionedCall-layer_normalization_8/StatefulPartitionedCall2f
1spectral_normalization_20/StatefulPartitionedCall1spectral_normalization_20/StatefulPartitionedCall2f
1spectral_normalization_21/StatefulPartitionedCall1spectral_normalization_21/StatefulPartitionedCall2f
1spectral_normalization_22/StatefulPartitionedCall1spectral_normalization_22/StatefulPartitionedCall2f
1spectral_normalization_23/StatefulPartitionedCall1spectral_normalization_23/StatefulPartitionedCall:X T
/
_output_shapes
:���������<
!
_user_specified_name	input_6
�L
�
"__inference__traced_restore_734920
file_prefixK
1assignvariableop_spectral_normalization_20_kernel:@C
1assignvariableop_1_spectral_normalization_20_sn_u:@<
.assignvariableop_2_layer_normalization_6_gamma:@;
-assignvariableop_3_layer_normalization_6_beta:@M
3assignvariableop_4_spectral_normalization_21_kernel:	@ C
1assignvariableop_5_spectral_normalization_21_sn_u: <
.assignvariableop_6_layer_normalization_7_gamma: ;
-assignvariableop_7_layer_normalization_7_beta: M
3assignvariableop_8_spectral_normalization_22_kernel: C
1assignvariableop_9_spectral_normalization_22_sn_u:=
/assignvariableop_10_layer_normalization_8_gamma:<
.assignvariableop_11_layer_normalization_8_beta:G
4assignvariableop_12_spectral_normalization_23_kernel:	�D
2assignvariableop_13_spectral_normalization_23_sn_u:@
2assignvariableop_14_spectral_normalization_20_bias:@@
2assignvariableop_15_spectral_normalization_21_bias: @
2assignvariableop_16_spectral_normalization_22_bias:@
2assignvariableop_17_spectral_normalization_23_bias:
identity_19��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/sn_u/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-2/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/sn_u/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-4/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/sn_u/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-6/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/sn_u/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp1assignvariableop_spectral_normalization_20_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp1assignvariableop_1_spectral_normalization_20_sn_uIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp.assignvariableop_2_layer_normalization_6_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp-assignvariableop_3_layer_normalization_6_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp3assignvariableop_4_spectral_normalization_21_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp1assignvariableop_5_spectral_normalization_21_sn_uIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp.assignvariableop_6_layer_normalization_7_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp-assignvariableop_7_layer_normalization_7_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp3assignvariableop_8_spectral_normalization_22_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp1assignvariableop_9_spectral_normalization_22_sn_uIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp/assignvariableop_10_layer_normalization_8_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp.assignvariableop_11_layer_normalization_8_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp4assignvariableop_12_spectral_normalization_23_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp2assignvariableop_13_spectral_normalization_23_sn_uIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp2assignvariableop_14_spectral_normalization_20_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp2assignvariableop_15_spectral_normalization_21_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp2assignvariableop_16_spectral_normalization_22_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp2assignvariableop_17_spectral_normalization_23_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_19IdentityIdentity_18:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_19Identity_19:output:0*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
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
�0
�	
__inference__traced_save_734856
file_prefix?
;savev2_spectral_normalization_20_kernel_read_readvariableop=
9savev2_spectral_normalization_20_sn_u_read_readvariableop:
6savev2_layer_normalization_6_gamma_read_readvariableop9
5savev2_layer_normalization_6_beta_read_readvariableop?
;savev2_spectral_normalization_21_kernel_read_readvariableop=
9savev2_spectral_normalization_21_sn_u_read_readvariableop:
6savev2_layer_normalization_7_gamma_read_readvariableop9
5savev2_layer_normalization_7_beta_read_readvariableop?
;savev2_spectral_normalization_22_kernel_read_readvariableop=
9savev2_spectral_normalization_22_sn_u_read_readvariableop:
6savev2_layer_normalization_8_gamma_read_readvariableop9
5savev2_layer_normalization_8_beta_read_readvariableop?
;savev2_spectral_normalization_23_kernel_read_readvariableop=
9savev2_spectral_normalization_23_sn_u_read_readvariableop=
9savev2_spectral_normalization_20_bias_read_readvariableop=
9savev2_spectral_normalization_21_bias_read_readvariableop=
9savev2_spectral_normalization_22_bias_read_readvariableop=
9savev2_spectral_normalization_23_bias_read_readvariableop
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
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/sn_u/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-2/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/sn_u/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-4/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/sn_u/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-6/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/sn_u/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B �	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0;savev2_spectral_normalization_20_kernel_read_readvariableop9savev2_spectral_normalization_20_sn_u_read_readvariableop6savev2_layer_normalization_6_gamma_read_readvariableop5savev2_layer_normalization_6_beta_read_readvariableop;savev2_spectral_normalization_21_kernel_read_readvariableop9savev2_spectral_normalization_21_sn_u_read_readvariableop6savev2_layer_normalization_7_gamma_read_readvariableop5savev2_layer_normalization_7_beta_read_readvariableop;savev2_spectral_normalization_22_kernel_read_readvariableop9savev2_spectral_normalization_22_sn_u_read_readvariableop6savev2_layer_normalization_8_gamma_read_readvariableop5savev2_layer_normalization_8_beta_read_readvariableop;savev2_spectral_normalization_23_kernel_read_readvariableop9savev2_spectral_normalization_23_sn_u_read_readvariableop9savev2_spectral_normalization_20_bias_read_readvariableop9savev2_spectral_normalization_21_bias_read_readvariableop9savev2_spectral_normalization_22_bias_read_readvariableop9savev2_spectral_normalization_23_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *!
dtypes
2�
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
_input_shapes�
�: :@:@:@:@:	@ : : : : ::::	�::@: ::: 2(
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

:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:	@ :$ 

_output_shapes

: : 

_output_shapes
: : 

_output_shapes
: :,	(
&
_output_shapes
: :$
 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	�:$ 

_output_shapes

:: 

_output_shapes
:@: 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
�
�
U__inference_spectral_normalization_21_layer_call_and_return_conditional_losses_732999

inputsB
(conv2d_19_conv2d_readvariableop_resource:	@ 7
)conv2d_19_biasadd_readvariableop_resource: 
identity�� conv2d_19/BiasAdd/ReadVariableOp�conv2d_19/Conv2D/ReadVariableOp�
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:	@ *
dtype0�
conv2d_19/Conv2DConv2Dinputs'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< *
paddingSAME*
strides
�
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< {
!conv2d_19/leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_19/BiasAdd:output:0*/
_output_shapes
:���������< �
IdentityIdentity/conv2d_19/leaky_re_lu_2/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������< �
NoOpNoOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<@: : 2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������<@
 
_user_specified_nameinputs
�
�
U__inference_spectral_normalization_21_layer_call_and_return_conditional_losses_734466

inputsB
(conv2d_19_conv2d_readvariableop_resource:	@ 7
)conv2d_19_biasadd_readvariableop_resource: 
identity�� conv2d_19/BiasAdd/ReadVariableOp�conv2d_19/Conv2D/ReadVariableOp�
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:	@ *
dtype0�
conv2d_19/Conv2DConv2Dinputs'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< *
paddingSAME*
strides
�
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< {
!conv2d_19/leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_19/BiasAdd:output:0*/
_output_shapes
:���������< �
IdentityIdentity/conv2d_19/leaky_re_lu_2/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������< �
NoOpNoOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<@: : 2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������<@
 
_user_specified_nameinputs
�
�
:__inference_spectral_normalization_23_layer_call_fn_734719

inputs
unknown:	�
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
GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_23_layer_call_and_return_conditional_losses_733156o
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
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
.__inference_discriminator_layer_call_fn_733792

inputs!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@#
	unknown_3:	@ 
	unknown_4: 
	unknown_5: 
	unknown_6: #
	unknown_7: 
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_discriminator_layer_call_and_return_conditional_losses_733163o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������<: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�7
�
U__inference_spectral_normalization_21_layer_call_and_return_conditional_losses_733393

inputs9
reshape_readvariableop_resource:	@ C
1spectral_normalize_matmul_readvariableop_resource: 7
)conv2d_19_biasadd_readvariableop_resource: 
identity��Reshape/ReadVariableOp� conv2d_19/BiasAdd/ReadVariableOp�conv2d_19/Conv2D/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOp~
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
conv2d_19/Conv2D/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
:	@ *
dtype0�
conv2d_19/Conv2DConv2Dinputs'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< *
paddingSAME*
strides
�
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< {
!conv2d_19/leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_19/BiasAdd:output:0*/
_output_shapes
:���������< �
IdentityIdentity/conv2d_19/leaky_re_lu_2/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������< �
NoOpNoOp^Reshape/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������<@: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:���������<@
 
_user_specified_nameinputs
�)
�
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_732982

inputs+
mul_4_readvariableop_resource:@)
add_readvariableop_resource:@
identity��add/ReadVariableOp�mul_4/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
mul_2Mul	mul_1:z:0strided_slice_2:output:0*
T0*
_output_shapes
: _
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_3Mulmul_3/x:output:0strided_slice_3:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackReshape/shape/0:output:0	mul_2:z:0	mul_3:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������@L
ones/packedPack	mul_2:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:���������M
zeros/packedPack	mul_2:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:���������H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB �
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������@:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:t
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*/
_output_shapes
:���������<@n
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes
:@*
dtype0x
mul_4MulReshape_1:output:0mul_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<@j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype0m
addAddV2	mul_4:z:0add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<@^
IdentityIdentityadd:z:0^NoOp*
T0*/
_output_shapes
:���������<@r
NoOpNoOp^add/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<@: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:W S
/
_output_shapes
:���������<@
 
_user_specified_nameinputs
�1
�
I__inference_discriminator_layer_call_and_return_conditional_losses_733163

inputs:
 spectral_normalization_20_732925:@.
 spectral_normalization_20_732927:@*
layer_normalization_6_732983:@*
layer_normalization_6_732985:@:
 spectral_normalization_21_733000:	@ .
 spectral_normalization_21_733002: *
layer_normalization_7_733058: *
layer_normalization_7_733060: :
 spectral_normalization_22_733075: .
 spectral_normalization_22_733077:*
layer_normalization_8_733133:*
layer_normalization_8_733135:3
 spectral_normalization_23_733157:	�.
 spectral_normalization_23_733159:
identity��-layer_normalization_6/StatefulPartitionedCall�-layer_normalization_7/StatefulPartitionedCall�-layer_normalization_8/StatefulPartitionedCall�1spectral_normalization_20/StatefulPartitionedCall�1spectral_normalization_21/StatefulPartitionedCall�1spectral_normalization_22/StatefulPartitionedCall�1spectral_normalization_23/StatefulPartitionedCall�
1spectral_normalization_20/StatefulPartitionedCallStatefulPartitionedCallinputs spectral_normalization_20_732925 spectral_normalization_20_732927*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_20_layer_call_and_return_conditional_losses_732924�
-layer_normalization_6/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_20/StatefulPartitionedCall:output:0layer_normalization_6_732983layer_normalization_6_732985*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_732982�
1spectral_normalization_21/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_6/StatefulPartitionedCall:output:0 spectral_normalization_21_733000 spectral_normalization_21_733002*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������< *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_21_layer_call_and_return_conditional_losses_732999�
-layer_normalization_7/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_21/StatefulPartitionedCall:output:0layer_normalization_7_733058layer_normalization_7_733060*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������< *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_733057�
1spectral_normalization_22/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_7/StatefulPartitionedCall:output:0 spectral_normalization_22_733075 spectral_normalization_22_733077*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_22_layer_call_and_return_conditional_losses_733074�
-layer_normalization_8/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_22/StatefulPartitionedCall:output:0layer_normalization_8_733133layer_normalization_8_733135*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_layer_normalization_8_layer_call_and_return_conditional_losses_733132�
flatten_2/PartitionedCallPartitionedCall6layer_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_733144�
1spectral_normalization_23/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0 spectral_normalization_23_733157 spectral_normalization_23_733159*
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
GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_23_layer_call_and_return_conditional_losses_733156�
IdentityIdentity:spectral_normalization_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^layer_normalization_6/StatefulPartitionedCall.^layer_normalization_7/StatefulPartitionedCall.^layer_normalization_8/StatefulPartitionedCall2^spectral_normalization_20/StatefulPartitionedCall2^spectral_normalization_21/StatefulPartitionedCall2^spectral_normalization_22/StatefulPartitionedCall2^spectral_normalization_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������<: : : : : : : : : : : : : : 2^
-layer_normalization_6/StatefulPartitionedCall-layer_normalization_6/StatefulPartitionedCall2^
-layer_normalization_7/StatefulPartitionedCall-layer_normalization_7/StatefulPartitionedCall2^
-layer_normalization_8/StatefulPartitionedCall-layer_normalization_8/StatefulPartitionedCall2f
1spectral_normalization_20/StatefulPartitionedCall1spectral_normalization_20/StatefulPartitionedCall2f
1spectral_normalization_21/StatefulPartitionedCall1spectral_normalization_21/StatefulPartitionedCall2f
1spectral_normalization_22/StatefulPartitionedCall1spectral_normalization_22/StatefulPartitionedCall2f
1spectral_normalization_23/StatefulPartitionedCall1spectral_normalization_23/StatefulPartitionedCall:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�7
�
U__inference_spectral_normalization_20_layer_call_and_return_conditional_losses_733464

inputs9
reshape_readvariableop_resource:@C
1spectral_normalize_matmul_readvariableop_resource:@7
)conv2d_18_biasadd_readvariableop_resource:@
identity��Reshape/ReadVariableOp� conv2d_18/BiasAdd/ReadVariableOp�conv2d_18/Conv2D/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOp~
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
conv2d_18/Conv2D/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
:@*
dtype0�
conv2d_18/Conv2DConv2Dinputs'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<@*
paddingSAME*
strides
�
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<@{
!conv2d_18/leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_18/BiasAdd:output:0*/
_output_shapes
:���������<@�
IdentityIdentity/conv2d_18/leaky_re_lu_2/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������<@�
NoOpNoOp^Reshape/ReadVariableOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������<: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
6__inference_layer_normalization_6_layer_call_fn_734383

inputs
unknown:@
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
:���������<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_732982w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������<@
 
_user_specified_nameinputs
�
a
E__inference_flatten_2_layer_call_and_return_conditional_losses_734710

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�
a
E__inference_flatten_2_layer_call_and_return_conditional_losses_733144

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
.__inference_discriminator_layer_call_fn_733636
input_6!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@#
	unknown_4:	@ 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: #
	unknown_9: 

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:	�

unknown_15:

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_discriminator_layer_call_and_return_conditional_losses_733556o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������<: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������<
!
_user_specified_name	input_6
�
�
.__inference_discriminator_layer_call_fn_733833

inputs!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@#
	unknown_4:	@ 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: #
	unknown_9: 

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:	�

unknown_15:

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_discriminator_layer_call_and_return_conditional_losses_733556o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������<: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
U__inference_spectral_normalization_20_layer_call_and_return_conditional_losses_732924

inputsB
(conv2d_18_conv2d_readvariableop_resource:@7
)conv2d_18_biasadd_readvariableop_resource:@
identity�� conv2d_18/BiasAdd/ReadVariableOp�conv2d_18/Conv2D/ReadVariableOp�
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d_18/Conv2DConv2Dinputs'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<@*
paddingSAME*
strides
�
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<@{
!conv2d_18/leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_18/BiasAdd:output:0*/
_output_shapes
:���������<@�
IdentityIdentity/conv2d_18/leaky_re_lu_2/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������<@�
NoOpNoOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<: : 2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�7
�
U__inference_spectral_normalization_21_layer_call_and_return_conditional_losses_734506

inputs9
reshape_readvariableop_resource:	@ C
1spectral_normalize_matmul_readvariableop_resource: 7
)conv2d_19_biasadd_readvariableop_resource: 
identity��Reshape/ReadVariableOp� conv2d_19/BiasAdd/ReadVariableOp�conv2d_19/Conv2D/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOp~
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
conv2d_19/Conv2D/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
:	@ *
dtype0�
conv2d_19/Conv2DConv2Dinputs'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< *
paddingSAME*
strides
�
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< {
!conv2d_19/leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_19/BiasAdd:output:0*/
_output_shapes
:���������< �
IdentityIdentity/conv2d_19/leaky_re_lu_2/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������< �
NoOpNoOp^Reshape/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������<@: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:���������<@
 
_user_specified_nameinputs
�
�
U__inference_spectral_normalization_23_layer_call_and_return_conditional_losses_733156

inputs9
&dense_2_matmul_readvariableop_resource:	�5
'dense_2_biasadd_readvariableop_resource:
identity��dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0y
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�)
�
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_734567

inputs+
mul_4_readvariableop_resource: )
add_readvariableop_resource: 
identity��add/ReadVariableOp�mul_4/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
mul_2Mul	mul_1:z:0strided_slice_2:output:0*
T0*
_output_shapes
: _
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_3Mulmul_3/x:output:0strided_slice_3:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackReshape/shape/0:output:0	mul_2:z:0	mul_3:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:��������� L
ones/packedPack	mul_2:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:���������M
zeros/packedPack	mul_2:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:���������H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB �
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:��������� :���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:t
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*/
_output_shapes
:���������< n
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes
: *
dtype0x
mul_4MulReshape_1:output:0mul_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0m
addAddV2	mul_4:z:0add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< ^
IdentityIdentityadd:z:0^NoOp*
T0*/
_output_shapes
:���������< r
NoOpNoOp^add/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������< : : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:W S
/
_output_shapes
:���������< 
 
_user_specified_nameinputs
�
�
U__inference_spectral_normalization_20_layer_call_and_return_conditional_losses_734334

inputsB
(conv2d_18_conv2d_readvariableop_resource:@7
)conv2d_18_biasadd_readvariableop_resource:@
identity�� conv2d_18/BiasAdd/ReadVariableOp�conv2d_18/Conv2D/ReadVariableOp�
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d_18/Conv2DConv2Dinputs'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<@*
paddingSAME*
strides
�
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<@{
!conv2d_18/leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_18/BiasAdd:output:0*/
_output_shapes
:���������<@�
IdentityIdentity/conv2d_18/leaky_re_lu_2/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������<@�
NoOpNoOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<: : 2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�)
�
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_733057

inputs+
mul_4_readvariableop_resource: )
add_readvariableop_resource: 
identity��add/ReadVariableOp�mul_4/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
mul_2Mul	mul_1:z:0strided_slice_2:output:0*
T0*
_output_shapes
: _
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_3Mulmul_3/x:output:0strided_slice_3:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackReshape/shape/0:output:0	mul_2:z:0	mul_3:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:��������� L
ones/packedPack	mul_2:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:���������M
zeros/packedPack	mul_2:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:���������H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB �
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:��������� :���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:t
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*/
_output_shapes
:���������< n
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes
: *
dtype0x
mul_4MulReshape_1:output:0mul_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0m
addAddV2	mul_4:z:0add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< ^
IdentityIdentityadd:z:0^NoOp*
T0*/
_output_shapes
:���������< r
NoOpNoOp^add/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������< : : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:W S
/
_output_shapes
:���������< 
 
_user_specified_nameinputs
�
�
.__inference_discriminator_layer_call_fn_733194
input_6!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@#
	unknown_3:	@ 
	unknown_4: 
	unknown_5: 
	unknown_6: #
	unknown_7: 
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_discriminator_layer_call_and_return_conditional_losses_733163o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������<: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������<
!
_user_specified_name	input_6
�5
�
U__inference_spectral_normalization_23_layer_call_and_return_conditional_losses_733245

inputs2
reshape_readvariableop_resource:	�C
1spectral_normalize_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identity��Reshape/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOpw
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:	�*
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   t
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	��
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes
:	�*
transpose_b(
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	�v
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
:	��
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes

:�
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:x
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

:
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:�
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	��
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes

:�
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
!spectral_normalize/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:	�*
dtype0�
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*
_output_shapes
:	�q
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�     �
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*
_output_shapes
:	��
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
dense_2/MatMul/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*
_output_shapes
:	�*
dtype0y
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^Reshape/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:����������: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�)
�
Q__inference_layer_normalization_8_layer_call_and_return_conditional_losses_734699

inputs+
mul_4_readvariableop_resource:)
add_readvariableop_resource:
identity��add/ReadVariableOp�mul_4/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
mul_2Mul	mul_1:z:0strided_slice_2:output:0*
T0*
_output_shapes
: _
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_3Mulmul_3/x:output:0strided_slice_3:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackReshape/shape/0:output:0	mul_2:z:0	mul_3:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������L
ones/packedPack	mul_2:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:���������M
zeros/packedPack	mul_2:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:���������H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB �
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:t
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*/
_output_shapes
:���������<n
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes
:*
dtype0x
mul_4MulReshape_1:output:0mul_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0m
addAddV2	mul_4:z:0add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<^
IdentityIdentityadd:z:0^NoOp*
T0*/
_output_shapes
:���������<r
NoOpNoOp^add/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
:__inference_spectral_normalization_20_layer_call_fn_734323

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
:���������<@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_20_layer_call_and_return_conditional_losses_733464w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������<: : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
:__inference_spectral_normalization_22_layer_call_fn_734576

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
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_22_layer_call_and_return_conditional_losses_733074w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������< : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������< 
 
_user_specified_nameinputs
�7
�
U__inference_spectral_normalization_22_layer_call_and_return_conditional_losses_734638

inputs9
reshape_readvariableop_resource: C
1spectral_normalize_matmul_readvariableop_resource:7
)conv2d_20_biasadd_readvariableop_resource:
identity��Reshape/ReadVariableOp� conv2d_20/BiasAdd/ReadVariableOp�conv2d_20/Conv2D/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOp~
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
conv2d_20/Conv2D/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
: *
dtype0�
conv2d_20/Conv2DConv2Dinputs'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<*
paddingSAME*
strides
�
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<{
!conv2d_20/leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_20/BiasAdd:output:0*/
_output_shapes
:���������<�
IdentityIdentity/conv2d_20/leaky_re_lu_2/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������<�
NoOpNoOp^Reshape/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������< : : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:���������< 
 
_user_specified_nameinputs
�
�
:__inference_spectral_normalization_21_layer_call_fn_734455

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
:���������< *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_21_layer_call_and_return_conditional_losses_733393w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������< `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������<@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������<@
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_733759
input_6!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@#
	unknown_3:	@ 
	unknown_4: 
	unknown_5: 
	unknown_6: #
	unknown_7: 
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_732906o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������<: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������<
!
_user_specified_name	input_6
�
�
:__inference_spectral_normalization_22_layer_call_fn_734587

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
:���������<*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_22_layer_call_and_return_conditional_losses_733322w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������< : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������< 
 
_user_specified_nameinputs
��
�
I__inference_discriminator_layer_call_and_return_conditional_losses_734303

inputsS
9spectral_normalization_20_reshape_readvariableop_resource:@]
Kspectral_normalization_20_spectral_normalize_matmul_readvariableop_resource:@Q
Cspectral_normalization_20_conv2d_18_biasadd_readvariableop_resource:@A
3layer_normalization_6_mul_4_readvariableop_resource:@?
1layer_normalization_6_add_readvariableop_resource:@S
9spectral_normalization_21_reshape_readvariableop_resource:	@ ]
Kspectral_normalization_21_spectral_normalize_matmul_readvariableop_resource: Q
Cspectral_normalization_21_conv2d_19_biasadd_readvariableop_resource: A
3layer_normalization_7_mul_4_readvariableop_resource: ?
1layer_normalization_7_add_readvariableop_resource: S
9spectral_normalization_22_reshape_readvariableop_resource: ]
Kspectral_normalization_22_spectral_normalize_matmul_readvariableop_resource:Q
Cspectral_normalization_22_conv2d_20_biasadd_readvariableop_resource:A
3layer_normalization_8_mul_4_readvariableop_resource:?
1layer_normalization_8_add_readvariableop_resource:L
9spectral_normalization_23_reshape_readvariableop_resource:	�]
Kspectral_normalization_23_spectral_normalize_matmul_readvariableop_resource:O
Aspectral_normalization_23_dense_2_biasadd_readvariableop_resource:
identity��(layer_normalization_6/add/ReadVariableOp�*layer_normalization_6/mul_4/ReadVariableOp�(layer_normalization_7/add/ReadVariableOp�*layer_normalization_7/mul_4/ReadVariableOp�(layer_normalization_8/add/ReadVariableOp�*layer_normalization_8/mul_4/ReadVariableOp�0spectral_normalization_20/Reshape/ReadVariableOp�:spectral_normalization_20/conv2d_18/BiasAdd/ReadVariableOp�9spectral_normalization_20/conv2d_18/Conv2D/ReadVariableOp�=spectral_normalization_20/spectral_normalize/AssignVariableOp�?spectral_normalization_20/spectral_normalize/AssignVariableOp_1�Bspectral_normalization_20/spectral_normalize/MatMul/ReadVariableOp�;spectral_normalization_20/spectral_normalize/ReadVariableOp�0spectral_normalization_21/Reshape/ReadVariableOp�:spectral_normalization_21/conv2d_19/BiasAdd/ReadVariableOp�9spectral_normalization_21/conv2d_19/Conv2D/ReadVariableOp�=spectral_normalization_21/spectral_normalize/AssignVariableOp�?spectral_normalization_21/spectral_normalize/AssignVariableOp_1�Bspectral_normalization_21/spectral_normalize/MatMul/ReadVariableOp�;spectral_normalization_21/spectral_normalize/ReadVariableOp�0spectral_normalization_22/Reshape/ReadVariableOp�:spectral_normalization_22/conv2d_20/BiasAdd/ReadVariableOp�9spectral_normalization_22/conv2d_20/Conv2D/ReadVariableOp�=spectral_normalization_22/spectral_normalize/AssignVariableOp�?spectral_normalization_22/spectral_normalize/AssignVariableOp_1�Bspectral_normalization_22/spectral_normalize/MatMul/ReadVariableOp�;spectral_normalization_22/spectral_normalize/ReadVariableOp�0spectral_normalization_23/Reshape/ReadVariableOp�8spectral_normalization_23/dense_2/BiasAdd/ReadVariableOp�7spectral_normalization_23/dense_2/MatMul/ReadVariableOp�=spectral_normalization_23/spectral_normalize/AssignVariableOp�?spectral_normalization_23/spectral_normalize/AssignVariableOp_1�Bspectral_normalization_23/spectral_normalize/MatMul/ReadVariableOp�;spectral_normalization_23/spectral_normalize/ReadVariableOp�
0spectral_normalization_20/Reshape/ReadVariableOpReadVariableOp9spectral_normalization_20_reshape_readvariableop_resource*&
_output_shapes
:@*
dtype0x
'spectral_normalization_20/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
!spectral_normalization_20/ReshapeReshape8spectral_normalization_20/Reshape/ReadVariableOp:value:00spectral_normalization_20/Reshape/shape:output:0*
T0*
_output_shapes

:H@�
Bspectral_normalization_20/spectral_normalize/MatMul/ReadVariableOpReadVariableOpKspectral_normalization_20_spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
3spectral_normalization_20/spectral_normalize/MatMulMatMulJspectral_normalization_20/spectral_normalize/MatMul/ReadVariableOp:value:0*spectral_normalization_20/Reshape:output:0*
T0*
_output_shapes

:H*
transpose_b(�
@spectral_normalization_20/spectral_normalize/l2_normalize/SquareSquare=spectral_normalization_20/spectral_normalize/MatMul:product:0*
T0*
_output_shapes

:H�
?spectral_normalization_20/spectral_normalize/l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
=spectral_normalization_20/spectral_normalize/l2_normalize/SumSumDspectral_normalization_20/spectral_normalize/l2_normalize/Square:y:0Hspectral_normalization_20/spectral_normalize/l2_normalize/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(�
Cspectral_normalization_20/spectral_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
Aspectral_normalization_20/spectral_normalize/l2_normalize/MaximumMaximumFspectral_normalization_20/spectral_normalize/l2_normalize/Sum:output:0Lspectral_normalization_20/spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:�
?spectral_normalization_20/spectral_normalize/l2_normalize/RsqrtRsqrtEspectral_normalization_20/spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:�
9spectral_normalization_20/spectral_normalize/l2_normalizeMul=spectral_normalization_20/spectral_normalize/MatMul:product:0Cspectral_normalization_20/spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes

:H�
5spectral_normalization_20/spectral_normalize/MatMul_1MatMul=spectral_normalization_20/spectral_normalize/l2_normalize:z:0*spectral_normalization_20/Reshape:output:0*
T0*
_output_shapes

:@�
Bspectral_normalization_20/spectral_normalize/l2_normalize_1/SquareSquare?spectral_normalization_20/spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:@�
Aspectral_normalization_20/spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
?spectral_normalization_20/spectral_normalize/l2_normalize_1/SumSumFspectral_normalization_20/spectral_normalize/l2_normalize_1/Square:y:0Jspectral_normalization_20/spectral_normalize/l2_normalize_1/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(�
Espectral_normalization_20/spectral_normalize/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
Cspectral_normalization_20/spectral_normalize/l2_normalize_1/MaximumMaximumHspectral_normalization_20/spectral_normalize/l2_normalize_1/Sum:output:0Nspectral_normalization_20/spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:�
Aspectral_normalization_20/spectral_normalize/l2_normalize_1/RsqrtRsqrtGspectral_normalization_20/spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:�
;spectral_normalization_20/spectral_normalize/l2_normalize_1Mul?spectral_normalization_20/spectral_normalize/MatMul_1:product:0Espectral_normalization_20/spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

:@�
9spectral_normalization_20/spectral_normalize/StopGradientStopGradient?spectral_normalization_20/spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:@�
;spectral_normalization_20/spectral_normalize/StopGradient_1StopGradient=spectral_normalization_20/spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes

:H�
5spectral_normalization_20/spectral_normalize/MatMul_2MatMulDspectral_normalization_20/spectral_normalize/StopGradient_1:output:0*spectral_normalization_20/Reshape:output:0*
T0*
_output_shapes

:@�
5spectral_normalization_20/spectral_normalize/MatMul_3MatMul?spectral_normalization_20/spectral_normalize/MatMul_2:product:0Bspectral_normalization_20/spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(�
=spectral_normalization_20/spectral_normalize/AssignVariableOpAssignVariableOpKspectral_normalization_20_spectral_normalize_matmul_readvariableop_resourceBspectral_normalization_20/spectral_normalize/StopGradient:output:0C^spectral_normalization_20/spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
;spectral_normalization_20/spectral_normalize/ReadVariableOpReadVariableOp9spectral_normalization_20_reshape_readvariableop_resource*&
_output_shapes
:@*
dtype0�
4spectral_normalization_20/spectral_normalize/truedivRealDivCspectral_normalization_20/spectral_normalize/ReadVariableOp:value:0?spectral_normalization_20/spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:@�
:spectral_normalization_20/spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
4spectral_normalization_20/spectral_normalize/ReshapeReshape8spectral_normalization_20/spectral_normalize/truediv:z:0Cspectral_normalization_20/spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:@�
?spectral_normalization_20/spectral_normalize/AssignVariableOp_1AssignVariableOp9spectral_normalization_20_reshape_readvariableop_resource=spectral_normalization_20/spectral_normalize/Reshape:output:01^spectral_normalization_20/Reshape/ReadVariableOp<^spectral_normalization_20/spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
9spectral_normalization_20/conv2d_18/Conv2D/ReadVariableOpReadVariableOp9spectral_normalization_20_reshape_readvariableop_resource@^spectral_normalization_20/spectral_normalize/AssignVariableOp_1*&
_output_shapes
:@*
dtype0�
*spectral_normalization_20/conv2d_18/Conv2DConv2DinputsAspectral_normalization_20/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<@*
paddingSAME*
strides
�
:spectral_normalization_20/conv2d_18/BiasAdd/ReadVariableOpReadVariableOpCspectral_normalization_20_conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
+spectral_normalization_20/conv2d_18/BiasAddBiasAdd3spectral_normalization_20/conv2d_18/Conv2D:output:0Bspectral_normalization_20/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<@�
;spectral_normalization_20/conv2d_18/leaky_re_lu_2/LeakyRelu	LeakyRelu4spectral_normalization_20/conv2d_18/BiasAdd:output:0*/
_output_shapes
:���������<@�
layer_normalization_6/ShapeShapeIspectral_normalization_20/conv2d_18/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:s
)layer_normalization_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_6/strided_sliceStridedSlice$layer_normalization_6/Shape:output:02layer_normalization_6/strided_slice/stack:output:04layer_normalization_6/strided_slice/stack_1:output:04layer_normalization_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_6/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_6/mulMul$layer_normalization_6/mul/x:output:0,layer_normalization_6/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_6/strided_slice_1StridedSlice$layer_normalization_6/Shape:output:04layer_normalization_6/strided_slice_1/stack:output:06layer_normalization_6/strided_slice_1/stack_1:output:06layer_normalization_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_6/mul_1Mullayer_normalization_6/mul:z:0.layer_normalization_6/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_6/strided_slice_2StridedSlice$layer_normalization_6/Shape:output:04layer_normalization_6/strided_slice_2/stack:output:06layer_normalization_6/strided_slice_2/stack_1:output:06layer_normalization_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_6/mul_2Mullayer_normalization_6/mul_1:z:0.layer_normalization_6/strided_slice_2:output:0*
T0*
_output_shapes
: u
+layer_normalization_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_6/strided_slice_3StridedSlice$layer_normalization_6/Shape:output:04layer_normalization_6/strided_slice_3/stack:output:06layer_normalization_6/strided_slice_3/stack_1:output:06layer_normalization_6/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_6/mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_6/mul_3Mul&layer_normalization_6/mul_3/x:output:0.layer_normalization_6/strided_slice_3:output:0*
T0*
_output_shapes
: g
%layer_normalization_6/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_6/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
#layer_normalization_6/Reshape/shapePack.layer_normalization_6/Reshape/shape/0:output:0layer_normalization_6/mul_2:z:0layer_normalization_6/mul_3:z:0.layer_normalization_6/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
layer_normalization_6/ReshapeReshapeIspectral_normalization_20/conv2d_18/leaky_re_lu_2/LeakyRelu:activations:0,layer_normalization_6/Reshape/shape:output:0*
T0*/
_output_shapes
:���������@x
!layer_normalization_6/ones/packedPacklayer_normalization_6/mul_2:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_6/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
layer_normalization_6/onesFill*layer_normalization_6/ones/packed:output:0)layer_normalization_6/ones/Const:output:0*
T0*#
_output_shapes
:���������y
"layer_normalization_6/zeros/packedPacklayer_normalization_6/mul_2:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
layer_normalization_6/zerosFill+layer_normalization_6/zeros/packed:output:0*layer_normalization_6/zeros/Const:output:0*
T0*#
_output_shapes
:���������^
layer_normalization_6/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
&layer_normalization_6/FusedBatchNormV3FusedBatchNormV3&layer_normalization_6/Reshape:output:0#layer_normalization_6/ones:output:0$layer_normalization_6/zeros:output:0$layer_normalization_6/Const:output:0&layer_normalization_6/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������@:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
layer_normalization_6/Reshape_1Reshape*layer_normalization_6/FusedBatchNormV3:y:0$layer_normalization_6/Shape:output:0*
T0*/
_output_shapes
:���������<@�
*layer_normalization_6/mul_4/ReadVariableOpReadVariableOp3layer_normalization_6_mul_4_readvariableop_resource*
_output_shapes
:@*
dtype0�
layer_normalization_6/mul_4Mul(layer_normalization_6/Reshape_1:output:02layer_normalization_6/mul_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<@�
(layer_normalization_6/add/ReadVariableOpReadVariableOp1layer_normalization_6_add_readvariableop_resource*
_output_shapes
:@*
dtype0�
layer_normalization_6/addAddV2layer_normalization_6/mul_4:z:00layer_normalization_6/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<@�
0spectral_normalization_21/Reshape/ReadVariableOpReadVariableOp9spectral_normalization_21_reshape_readvariableop_resource*&
_output_shapes
:	@ *
dtype0x
'spectral_normalization_21/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
!spectral_normalization_21/ReshapeReshape8spectral_normalization_21/Reshape/ReadVariableOp:value:00spectral_normalization_21/Reshape/shape:output:0*
T0*
_output_shapes
:	� �
Bspectral_normalization_21/spectral_normalize/MatMul/ReadVariableOpReadVariableOpKspectral_normalization_21_spectral_normalize_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
3spectral_normalization_21/spectral_normalize/MatMulMatMulJspectral_normalization_21/spectral_normalize/MatMul/ReadVariableOp:value:0*spectral_normalization_21/Reshape:output:0*
T0*
_output_shapes
:	�*
transpose_b(�
@spectral_normalization_21/spectral_normalize/l2_normalize/SquareSquare=spectral_normalization_21/spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	��
?spectral_normalization_21/spectral_normalize/l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
=spectral_normalization_21/spectral_normalize/l2_normalize/SumSumDspectral_normalization_21/spectral_normalize/l2_normalize/Square:y:0Hspectral_normalization_21/spectral_normalize/l2_normalize/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(�
Cspectral_normalization_21/spectral_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
Aspectral_normalization_21/spectral_normalize/l2_normalize/MaximumMaximumFspectral_normalization_21/spectral_normalize/l2_normalize/Sum:output:0Lspectral_normalization_21/spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:�
?spectral_normalization_21/spectral_normalize/l2_normalize/RsqrtRsqrtEspectral_normalization_21/spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:�
9spectral_normalization_21/spectral_normalize/l2_normalizeMul=spectral_normalization_21/spectral_normalize/MatMul:product:0Cspectral_normalization_21/spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes
:	��
5spectral_normalization_21/spectral_normalize/MatMul_1MatMul=spectral_normalization_21/spectral_normalize/l2_normalize:z:0*spectral_normalization_21/Reshape:output:0*
T0*
_output_shapes

: �
Bspectral_normalization_21/spectral_normalize/l2_normalize_1/SquareSquare?spectral_normalization_21/spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

: �
Aspectral_normalization_21/spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
?spectral_normalization_21/spectral_normalize/l2_normalize_1/SumSumFspectral_normalization_21/spectral_normalize/l2_normalize_1/Square:y:0Jspectral_normalization_21/spectral_normalize/l2_normalize_1/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(�
Espectral_normalization_21/spectral_normalize/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
Cspectral_normalization_21/spectral_normalize/l2_normalize_1/MaximumMaximumHspectral_normalization_21/spectral_normalize/l2_normalize_1/Sum:output:0Nspectral_normalization_21/spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:�
Aspectral_normalization_21/spectral_normalize/l2_normalize_1/RsqrtRsqrtGspectral_normalization_21/spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:�
;spectral_normalization_21/spectral_normalize/l2_normalize_1Mul?spectral_normalization_21/spectral_normalize/MatMul_1:product:0Espectral_normalization_21/spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

: �
9spectral_normalization_21/spectral_normalize/StopGradientStopGradient?spectral_normalization_21/spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

: �
;spectral_normalization_21/spectral_normalize/StopGradient_1StopGradient=spectral_normalization_21/spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	��
5spectral_normalization_21/spectral_normalize/MatMul_2MatMulDspectral_normalization_21/spectral_normalize/StopGradient_1:output:0*spectral_normalization_21/Reshape:output:0*
T0*
_output_shapes

: �
5spectral_normalization_21/spectral_normalize/MatMul_3MatMul?spectral_normalization_21/spectral_normalize/MatMul_2:product:0Bspectral_normalization_21/spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(�
=spectral_normalization_21/spectral_normalize/AssignVariableOpAssignVariableOpKspectral_normalization_21_spectral_normalize_matmul_readvariableop_resourceBspectral_normalization_21/spectral_normalize/StopGradient:output:0C^spectral_normalization_21/spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
;spectral_normalization_21/spectral_normalize/ReadVariableOpReadVariableOp9spectral_normalization_21_reshape_readvariableop_resource*&
_output_shapes
:	@ *
dtype0�
4spectral_normalization_21/spectral_normalize/truedivRealDivCspectral_normalization_21/spectral_normalize/ReadVariableOp:value:0?spectral_normalization_21/spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:	@ �
:spectral_normalization_21/spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   	   @       �
4spectral_normalization_21/spectral_normalize/ReshapeReshape8spectral_normalization_21/spectral_normalize/truediv:z:0Cspectral_normalization_21/spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:	@ �
?spectral_normalization_21/spectral_normalize/AssignVariableOp_1AssignVariableOp9spectral_normalization_21_reshape_readvariableop_resource=spectral_normalization_21/spectral_normalize/Reshape:output:01^spectral_normalization_21/Reshape/ReadVariableOp<^spectral_normalization_21/spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
9spectral_normalization_21/conv2d_19/Conv2D/ReadVariableOpReadVariableOp9spectral_normalization_21_reshape_readvariableop_resource@^spectral_normalization_21/spectral_normalize/AssignVariableOp_1*&
_output_shapes
:	@ *
dtype0�
*spectral_normalization_21/conv2d_19/Conv2DConv2Dlayer_normalization_6/add:z:0Aspectral_normalization_21/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< *
paddingSAME*
strides
�
:spectral_normalization_21/conv2d_19/BiasAdd/ReadVariableOpReadVariableOpCspectral_normalization_21_conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
+spectral_normalization_21/conv2d_19/BiasAddBiasAdd3spectral_normalization_21/conv2d_19/Conv2D:output:0Bspectral_normalization_21/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< �
;spectral_normalization_21/conv2d_19/leaky_re_lu_2/LeakyRelu	LeakyRelu4spectral_normalization_21/conv2d_19/BiasAdd:output:0*/
_output_shapes
:���������< �
layer_normalization_7/ShapeShapeIspectral_normalization_21/conv2d_19/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:s
)layer_normalization_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_7/strided_sliceStridedSlice$layer_normalization_7/Shape:output:02layer_normalization_7/strided_slice/stack:output:04layer_normalization_7/strided_slice/stack_1:output:04layer_normalization_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_7/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_7/mulMul$layer_normalization_7/mul/x:output:0,layer_normalization_7/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_7/strided_slice_1StridedSlice$layer_normalization_7/Shape:output:04layer_normalization_7/strided_slice_1/stack:output:06layer_normalization_7/strided_slice_1/stack_1:output:06layer_normalization_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_7/mul_1Mullayer_normalization_7/mul:z:0.layer_normalization_7/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_7/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_7/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_7/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_7/strided_slice_2StridedSlice$layer_normalization_7/Shape:output:04layer_normalization_7/strided_slice_2/stack:output:06layer_normalization_7/strided_slice_2/stack_1:output:06layer_normalization_7/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_7/mul_2Mullayer_normalization_7/mul_1:z:0.layer_normalization_7/strided_slice_2:output:0*
T0*
_output_shapes
: u
+layer_normalization_7/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_7/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_7/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_7/strided_slice_3StridedSlice$layer_normalization_7/Shape:output:04layer_normalization_7/strided_slice_3/stack:output:06layer_normalization_7/strided_slice_3/stack_1:output:06layer_normalization_7/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_7/mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_7/mul_3Mul&layer_normalization_7/mul_3/x:output:0.layer_normalization_7/strided_slice_3:output:0*
T0*
_output_shapes
: g
%layer_normalization_7/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_7/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
#layer_normalization_7/Reshape/shapePack.layer_normalization_7/Reshape/shape/0:output:0layer_normalization_7/mul_2:z:0layer_normalization_7/mul_3:z:0.layer_normalization_7/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
layer_normalization_7/ReshapeReshapeIspectral_normalization_21/conv2d_19/leaky_re_lu_2/LeakyRelu:activations:0,layer_normalization_7/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� x
!layer_normalization_7/ones/packedPacklayer_normalization_7/mul_2:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_7/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
layer_normalization_7/onesFill*layer_normalization_7/ones/packed:output:0)layer_normalization_7/ones/Const:output:0*
T0*#
_output_shapes
:���������y
"layer_normalization_7/zeros/packedPacklayer_normalization_7/mul_2:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
layer_normalization_7/zerosFill+layer_normalization_7/zeros/packed:output:0*layer_normalization_7/zeros/Const:output:0*
T0*#
_output_shapes
:���������^
layer_normalization_7/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
&layer_normalization_7/FusedBatchNormV3FusedBatchNormV3&layer_normalization_7/Reshape:output:0#layer_normalization_7/ones:output:0$layer_normalization_7/zeros:output:0$layer_normalization_7/Const:output:0&layer_normalization_7/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:��������� :���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
layer_normalization_7/Reshape_1Reshape*layer_normalization_7/FusedBatchNormV3:y:0$layer_normalization_7/Shape:output:0*
T0*/
_output_shapes
:���������< �
*layer_normalization_7/mul_4/ReadVariableOpReadVariableOp3layer_normalization_7_mul_4_readvariableop_resource*
_output_shapes
: *
dtype0�
layer_normalization_7/mul_4Mul(layer_normalization_7/Reshape_1:output:02layer_normalization_7/mul_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< �
(layer_normalization_7/add/ReadVariableOpReadVariableOp1layer_normalization_7_add_readvariableop_resource*
_output_shapes
: *
dtype0�
layer_normalization_7/addAddV2layer_normalization_7/mul_4:z:00layer_normalization_7/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< �
0spectral_normalization_22/Reshape/ReadVariableOpReadVariableOp9spectral_normalization_22_reshape_readvariableop_resource*&
_output_shapes
: *
dtype0x
'spectral_normalization_22/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
!spectral_normalization_22/ReshapeReshape8spectral_normalization_22/Reshape/ReadVariableOp:value:00spectral_normalization_22/Reshape/shape:output:0*
T0*
_output_shapes

:`�
Bspectral_normalization_22/spectral_normalize/MatMul/ReadVariableOpReadVariableOpKspectral_normalization_22_spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
3spectral_normalization_22/spectral_normalize/MatMulMatMulJspectral_normalization_22/spectral_normalize/MatMul/ReadVariableOp:value:0*spectral_normalization_22/Reshape:output:0*
T0*
_output_shapes

:`*
transpose_b(�
@spectral_normalization_22/spectral_normalize/l2_normalize/SquareSquare=spectral_normalization_22/spectral_normalize/MatMul:product:0*
T0*
_output_shapes

:`�
?spectral_normalization_22/spectral_normalize/l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
=spectral_normalization_22/spectral_normalize/l2_normalize/SumSumDspectral_normalization_22/spectral_normalize/l2_normalize/Square:y:0Hspectral_normalization_22/spectral_normalize/l2_normalize/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(�
Cspectral_normalization_22/spectral_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
Aspectral_normalization_22/spectral_normalize/l2_normalize/MaximumMaximumFspectral_normalization_22/spectral_normalize/l2_normalize/Sum:output:0Lspectral_normalization_22/spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:�
?spectral_normalization_22/spectral_normalize/l2_normalize/RsqrtRsqrtEspectral_normalization_22/spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:�
9spectral_normalization_22/spectral_normalize/l2_normalizeMul=spectral_normalization_22/spectral_normalize/MatMul:product:0Cspectral_normalization_22/spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes

:`�
5spectral_normalization_22/spectral_normalize/MatMul_1MatMul=spectral_normalization_22/spectral_normalize/l2_normalize:z:0*spectral_normalization_22/Reshape:output:0*
T0*
_output_shapes

:�
Bspectral_normalization_22/spectral_normalize/l2_normalize_1/SquareSquare?spectral_normalization_22/spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:�
Aspectral_normalization_22/spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
?spectral_normalization_22/spectral_normalize/l2_normalize_1/SumSumFspectral_normalization_22/spectral_normalize/l2_normalize_1/Square:y:0Jspectral_normalization_22/spectral_normalize/l2_normalize_1/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(�
Espectral_normalization_22/spectral_normalize/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
Cspectral_normalization_22/spectral_normalize/l2_normalize_1/MaximumMaximumHspectral_normalization_22/spectral_normalize/l2_normalize_1/Sum:output:0Nspectral_normalization_22/spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:�
Aspectral_normalization_22/spectral_normalize/l2_normalize_1/RsqrtRsqrtGspectral_normalization_22/spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:�
;spectral_normalization_22/spectral_normalize/l2_normalize_1Mul?spectral_normalization_22/spectral_normalize/MatMul_1:product:0Espectral_normalization_22/spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

:�
9spectral_normalization_22/spectral_normalize/StopGradientStopGradient?spectral_normalization_22/spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:�
;spectral_normalization_22/spectral_normalize/StopGradient_1StopGradient=spectral_normalization_22/spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes

:`�
5spectral_normalization_22/spectral_normalize/MatMul_2MatMulDspectral_normalization_22/spectral_normalize/StopGradient_1:output:0*spectral_normalization_22/Reshape:output:0*
T0*
_output_shapes

:�
5spectral_normalization_22/spectral_normalize/MatMul_3MatMul?spectral_normalization_22/spectral_normalize/MatMul_2:product:0Bspectral_normalization_22/spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(�
=spectral_normalization_22/spectral_normalize/AssignVariableOpAssignVariableOpKspectral_normalization_22_spectral_normalize_matmul_readvariableop_resourceBspectral_normalization_22/spectral_normalize/StopGradient:output:0C^spectral_normalization_22/spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
;spectral_normalization_22/spectral_normalize/ReadVariableOpReadVariableOp9spectral_normalization_22_reshape_readvariableop_resource*&
_output_shapes
: *
dtype0�
4spectral_normalization_22/spectral_normalize/truedivRealDivCspectral_normalization_22/spectral_normalize/ReadVariableOp:value:0?spectral_normalization_22/spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
: �
:spectral_normalization_22/spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
4spectral_normalization_22/spectral_normalize/ReshapeReshape8spectral_normalization_22/spectral_normalize/truediv:z:0Cspectral_normalization_22/spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
: �
?spectral_normalization_22/spectral_normalize/AssignVariableOp_1AssignVariableOp9spectral_normalization_22_reshape_readvariableop_resource=spectral_normalization_22/spectral_normalize/Reshape:output:01^spectral_normalization_22/Reshape/ReadVariableOp<^spectral_normalization_22/spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
9spectral_normalization_22/conv2d_20/Conv2D/ReadVariableOpReadVariableOp9spectral_normalization_22_reshape_readvariableop_resource@^spectral_normalization_22/spectral_normalize/AssignVariableOp_1*&
_output_shapes
: *
dtype0�
*spectral_normalization_22/conv2d_20/Conv2DConv2Dlayer_normalization_7/add:z:0Aspectral_normalization_22/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<*
paddingSAME*
strides
�
:spectral_normalization_22/conv2d_20/BiasAdd/ReadVariableOpReadVariableOpCspectral_normalization_22_conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
+spectral_normalization_22/conv2d_20/BiasAddBiasAdd3spectral_normalization_22/conv2d_20/Conv2D:output:0Bspectral_normalization_22/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<�
;spectral_normalization_22/conv2d_20/leaky_re_lu_2/LeakyRelu	LeakyRelu4spectral_normalization_22/conv2d_20/BiasAdd:output:0*/
_output_shapes
:���������<�
layer_normalization_8/ShapeShapeIspectral_normalization_22/conv2d_20/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:s
)layer_normalization_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_8/strided_sliceStridedSlice$layer_normalization_8/Shape:output:02layer_normalization_8/strided_slice/stack:output:04layer_normalization_8/strided_slice/stack_1:output:04layer_normalization_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_8/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_8/mulMul$layer_normalization_8/mul/x:output:0,layer_normalization_8/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_8/strided_slice_1StridedSlice$layer_normalization_8/Shape:output:04layer_normalization_8/strided_slice_1/stack:output:06layer_normalization_8/strided_slice_1/stack_1:output:06layer_normalization_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_8/mul_1Mullayer_normalization_8/mul:z:0.layer_normalization_8/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_8/strided_slice_2StridedSlice$layer_normalization_8/Shape:output:04layer_normalization_8/strided_slice_2/stack:output:06layer_normalization_8/strided_slice_2/stack_1:output:06layer_normalization_8/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_8/mul_2Mullayer_normalization_8/mul_1:z:0.layer_normalization_8/strided_slice_2:output:0*
T0*
_output_shapes
: u
+layer_normalization_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_8/strided_slice_3StridedSlice$layer_normalization_8/Shape:output:04layer_normalization_8/strided_slice_3/stack:output:06layer_normalization_8/strided_slice_3/stack_1:output:06layer_normalization_8/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_8/mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_8/mul_3Mul&layer_normalization_8/mul_3/x:output:0.layer_normalization_8/strided_slice_3:output:0*
T0*
_output_shapes
: g
%layer_normalization_8/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
#layer_normalization_8/Reshape/shapePack.layer_normalization_8/Reshape/shape/0:output:0layer_normalization_8/mul_2:z:0layer_normalization_8/mul_3:z:0.layer_normalization_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
layer_normalization_8/ReshapeReshapeIspectral_normalization_22/conv2d_20/leaky_re_lu_2/LeakyRelu:activations:0,layer_normalization_8/Reshape/shape:output:0*
T0*/
_output_shapes
:���������x
!layer_normalization_8/ones/packedPacklayer_normalization_8/mul_2:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_8/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
layer_normalization_8/onesFill*layer_normalization_8/ones/packed:output:0)layer_normalization_8/ones/Const:output:0*
T0*#
_output_shapes
:���������y
"layer_normalization_8/zeros/packedPacklayer_normalization_8/mul_2:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
layer_normalization_8/zerosFill+layer_normalization_8/zeros/packed:output:0*layer_normalization_8/zeros/Const:output:0*
T0*#
_output_shapes
:���������^
layer_normalization_8/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
&layer_normalization_8/FusedBatchNormV3FusedBatchNormV3&layer_normalization_8/Reshape:output:0#layer_normalization_8/ones:output:0$layer_normalization_8/zeros:output:0$layer_normalization_8/Const:output:0&layer_normalization_8/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
layer_normalization_8/Reshape_1Reshape*layer_normalization_8/FusedBatchNormV3:y:0$layer_normalization_8/Shape:output:0*
T0*/
_output_shapes
:���������<�
*layer_normalization_8/mul_4/ReadVariableOpReadVariableOp3layer_normalization_8_mul_4_readvariableop_resource*
_output_shapes
:*
dtype0�
layer_normalization_8/mul_4Mul(layer_normalization_8/Reshape_1:output:02layer_normalization_8/mul_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<�
(layer_normalization_8/add/ReadVariableOpReadVariableOp1layer_normalization_8_add_readvariableop_resource*
_output_shapes
:*
dtype0�
layer_normalization_8/addAddV2layer_normalization_8/mul_4:z:00layer_normalization_8/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  �
flatten_2/ReshapeReshapelayer_normalization_8/add:z:0flatten_2/Const:output:0*
T0*(
_output_shapes
:�����������
0spectral_normalization_23/Reshape/ReadVariableOpReadVariableOp9spectral_normalization_23_reshape_readvariableop_resource*
_output_shapes
:	�*
dtype0x
'spectral_normalization_23/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
!spectral_normalization_23/ReshapeReshape8spectral_normalization_23/Reshape/ReadVariableOp:value:00spectral_normalization_23/Reshape/shape:output:0*
T0*
_output_shapes
:	��
Bspectral_normalization_23/spectral_normalize/MatMul/ReadVariableOpReadVariableOpKspectral_normalization_23_spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
3spectral_normalization_23/spectral_normalize/MatMulMatMulJspectral_normalization_23/spectral_normalize/MatMul/ReadVariableOp:value:0*spectral_normalization_23/Reshape:output:0*
T0*
_output_shapes
:	�*
transpose_b(�
@spectral_normalization_23/spectral_normalize/l2_normalize/SquareSquare=spectral_normalization_23/spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	��
?spectral_normalization_23/spectral_normalize/l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
=spectral_normalization_23/spectral_normalize/l2_normalize/SumSumDspectral_normalization_23/spectral_normalize/l2_normalize/Square:y:0Hspectral_normalization_23/spectral_normalize/l2_normalize/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(�
Cspectral_normalization_23/spectral_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
Aspectral_normalization_23/spectral_normalize/l2_normalize/MaximumMaximumFspectral_normalization_23/spectral_normalize/l2_normalize/Sum:output:0Lspectral_normalization_23/spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:�
?spectral_normalization_23/spectral_normalize/l2_normalize/RsqrtRsqrtEspectral_normalization_23/spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:�
9spectral_normalization_23/spectral_normalize/l2_normalizeMul=spectral_normalization_23/spectral_normalize/MatMul:product:0Cspectral_normalization_23/spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes
:	��
5spectral_normalization_23/spectral_normalize/MatMul_1MatMul=spectral_normalization_23/spectral_normalize/l2_normalize:z:0*spectral_normalization_23/Reshape:output:0*
T0*
_output_shapes

:�
Bspectral_normalization_23/spectral_normalize/l2_normalize_1/SquareSquare?spectral_normalization_23/spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:�
Aspectral_normalization_23/spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
?spectral_normalization_23/spectral_normalize/l2_normalize_1/SumSumFspectral_normalization_23/spectral_normalize/l2_normalize_1/Square:y:0Jspectral_normalization_23/spectral_normalize/l2_normalize_1/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(�
Espectral_normalization_23/spectral_normalize/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
Cspectral_normalization_23/spectral_normalize/l2_normalize_1/MaximumMaximumHspectral_normalization_23/spectral_normalize/l2_normalize_1/Sum:output:0Nspectral_normalization_23/spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:�
Aspectral_normalization_23/spectral_normalize/l2_normalize_1/RsqrtRsqrtGspectral_normalization_23/spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:�
;spectral_normalization_23/spectral_normalize/l2_normalize_1Mul?spectral_normalization_23/spectral_normalize/MatMul_1:product:0Espectral_normalization_23/spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

:�
9spectral_normalization_23/spectral_normalize/StopGradientStopGradient?spectral_normalization_23/spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:�
;spectral_normalization_23/spectral_normalize/StopGradient_1StopGradient=spectral_normalization_23/spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	��
5spectral_normalization_23/spectral_normalize/MatMul_2MatMulDspectral_normalization_23/spectral_normalize/StopGradient_1:output:0*spectral_normalization_23/Reshape:output:0*
T0*
_output_shapes

:�
5spectral_normalization_23/spectral_normalize/MatMul_3MatMul?spectral_normalization_23/spectral_normalize/MatMul_2:product:0Bspectral_normalization_23/spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(�
=spectral_normalization_23/spectral_normalize/AssignVariableOpAssignVariableOpKspectral_normalization_23_spectral_normalize_matmul_readvariableop_resourceBspectral_normalization_23/spectral_normalize/StopGradient:output:0C^spectral_normalization_23/spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
;spectral_normalization_23/spectral_normalize/ReadVariableOpReadVariableOp9spectral_normalization_23_reshape_readvariableop_resource*
_output_shapes
:	�*
dtype0�
4spectral_normalization_23/spectral_normalize/truedivRealDivCspectral_normalization_23/spectral_normalize/ReadVariableOp:value:0?spectral_normalization_23/spectral_normalize/MatMul_3:product:0*
T0*
_output_shapes
:	��
:spectral_normalization_23/spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�     �
4spectral_normalization_23/spectral_normalize/ReshapeReshape8spectral_normalization_23/spectral_normalize/truediv:z:0Cspectral_normalization_23/spectral_normalize/Reshape/shape:output:0*
T0*
_output_shapes
:	��
?spectral_normalization_23/spectral_normalize/AssignVariableOp_1AssignVariableOp9spectral_normalization_23_reshape_readvariableop_resource=spectral_normalization_23/spectral_normalize/Reshape:output:01^spectral_normalization_23/Reshape/ReadVariableOp<^spectral_normalization_23/spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
7spectral_normalization_23/dense_2/MatMul/ReadVariableOpReadVariableOp9spectral_normalization_23_reshape_readvariableop_resource@^spectral_normalization_23/spectral_normalize/AssignVariableOp_1*
_output_shapes
:	�*
dtype0�
(spectral_normalization_23/dense_2/MatMulMatMulflatten_2/Reshape:output:0?spectral_normalization_23/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
8spectral_normalization_23/dense_2/BiasAdd/ReadVariableOpReadVariableOpAspectral_normalization_23_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
)spectral_normalization_23/dense_2/BiasAddBiasAdd2spectral_normalization_23/dense_2/MatMul:product:0@spectral_normalization_23/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
IdentityIdentity2spectral_normalization_23/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp)^layer_normalization_6/add/ReadVariableOp+^layer_normalization_6/mul_4/ReadVariableOp)^layer_normalization_7/add/ReadVariableOp+^layer_normalization_7/mul_4/ReadVariableOp)^layer_normalization_8/add/ReadVariableOp+^layer_normalization_8/mul_4/ReadVariableOp1^spectral_normalization_20/Reshape/ReadVariableOp;^spectral_normalization_20/conv2d_18/BiasAdd/ReadVariableOp:^spectral_normalization_20/conv2d_18/Conv2D/ReadVariableOp>^spectral_normalization_20/spectral_normalize/AssignVariableOp@^spectral_normalization_20/spectral_normalize/AssignVariableOp_1C^spectral_normalization_20/spectral_normalize/MatMul/ReadVariableOp<^spectral_normalization_20/spectral_normalize/ReadVariableOp1^spectral_normalization_21/Reshape/ReadVariableOp;^spectral_normalization_21/conv2d_19/BiasAdd/ReadVariableOp:^spectral_normalization_21/conv2d_19/Conv2D/ReadVariableOp>^spectral_normalization_21/spectral_normalize/AssignVariableOp@^spectral_normalization_21/spectral_normalize/AssignVariableOp_1C^spectral_normalization_21/spectral_normalize/MatMul/ReadVariableOp<^spectral_normalization_21/spectral_normalize/ReadVariableOp1^spectral_normalization_22/Reshape/ReadVariableOp;^spectral_normalization_22/conv2d_20/BiasAdd/ReadVariableOp:^spectral_normalization_22/conv2d_20/Conv2D/ReadVariableOp>^spectral_normalization_22/spectral_normalize/AssignVariableOp@^spectral_normalization_22/spectral_normalize/AssignVariableOp_1C^spectral_normalization_22/spectral_normalize/MatMul/ReadVariableOp<^spectral_normalization_22/spectral_normalize/ReadVariableOp1^spectral_normalization_23/Reshape/ReadVariableOp9^spectral_normalization_23/dense_2/BiasAdd/ReadVariableOp8^spectral_normalization_23/dense_2/MatMul/ReadVariableOp>^spectral_normalization_23/spectral_normalize/AssignVariableOp@^spectral_normalization_23/spectral_normalize/AssignVariableOp_1C^spectral_normalization_23/spectral_normalize/MatMul/ReadVariableOp<^spectral_normalization_23/spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������<: : : : : : : : : : : : : : : : : : 2T
(layer_normalization_6/add/ReadVariableOp(layer_normalization_6/add/ReadVariableOp2X
*layer_normalization_6/mul_4/ReadVariableOp*layer_normalization_6/mul_4/ReadVariableOp2T
(layer_normalization_7/add/ReadVariableOp(layer_normalization_7/add/ReadVariableOp2X
*layer_normalization_7/mul_4/ReadVariableOp*layer_normalization_7/mul_4/ReadVariableOp2T
(layer_normalization_8/add/ReadVariableOp(layer_normalization_8/add/ReadVariableOp2X
*layer_normalization_8/mul_4/ReadVariableOp*layer_normalization_8/mul_4/ReadVariableOp2d
0spectral_normalization_20/Reshape/ReadVariableOp0spectral_normalization_20/Reshape/ReadVariableOp2x
:spectral_normalization_20/conv2d_18/BiasAdd/ReadVariableOp:spectral_normalization_20/conv2d_18/BiasAdd/ReadVariableOp2v
9spectral_normalization_20/conv2d_18/Conv2D/ReadVariableOp9spectral_normalization_20/conv2d_18/Conv2D/ReadVariableOp2~
=spectral_normalization_20/spectral_normalize/AssignVariableOp=spectral_normalization_20/spectral_normalize/AssignVariableOp2�
?spectral_normalization_20/spectral_normalize/AssignVariableOp_1?spectral_normalization_20/spectral_normalize/AssignVariableOp_12�
Bspectral_normalization_20/spectral_normalize/MatMul/ReadVariableOpBspectral_normalization_20/spectral_normalize/MatMul/ReadVariableOp2z
;spectral_normalization_20/spectral_normalize/ReadVariableOp;spectral_normalization_20/spectral_normalize/ReadVariableOp2d
0spectral_normalization_21/Reshape/ReadVariableOp0spectral_normalization_21/Reshape/ReadVariableOp2x
:spectral_normalization_21/conv2d_19/BiasAdd/ReadVariableOp:spectral_normalization_21/conv2d_19/BiasAdd/ReadVariableOp2v
9spectral_normalization_21/conv2d_19/Conv2D/ReadVariableOp9spectral_normalization_21/conv2d_19/Conv2D/ReadVariableOp2~
=spectral_normalization_21/spectral_normalize/AssignVariableOp=spectral_normalization_21/spectral_normalize/AssignVariableOp2�
?spectral_normalization_21/spectral_normalize/AssignVariableOp_1?spectral_normalization_21/spectral_normalize/AssignVariableOp_12�
Bspectral_normalization_21/spectral_normalize/MatMul/ReadVariableOpBspectral_normalization_21/spectral_normalize/MatMul/ReadVariableOp2z
;spectral_normalization_21/spectral_normalize/ReadVariableOp;spectral_normalization_21/spectral_normalize/ReadVariableOp2d
0spectral_normalization_22/Reshape/ReadVariableOp0spectral_normalization_22/Reshape/ReadVariableOp2x
:spectral_normalization_22/conv2d_20/BiasAdd/ReadVariableOp:spectral_normalization_22/conv2d_20/BiasAdd/ReadVariableOp2v
9spectral_normalization_22/conv2d_20/Conv2D/ReadVariableOp9spectral_normalization_22/conv2d_20/Conv2D/ReadVariableOp2~
=spectral_normalization_22/spectral_normalize/AssignVariableOp=spectral_normalization_22/spectral_normalize/AssignVariableOp2�
?spectral_normalization_22/spectral_normalize/AssignVariableOp_1?spectral_normalization_22/spectral_normalize/AssignVariableOp_12�
Bspectral_normalization_22/spectral_normalize/MatMul/ReadVariableOpBspectral_normalization_22/spectral_normalize/MatMul/ReadVariableOp2z
;spectral_normalization_22/spectral_normalize/ReadVariableOp;spectral_normalization_22/spectral_normalize/ReadVariableOp2d
0spectral_normalization_23/Reshape/ReadVariableOp0spectral_normalization_23/Reshape/ReadVariableOp2t
8spectral_normalization_23/dense_2/BiasAdd/ReadVariableOp8spectral_normalization_23/dense_2/BiasAdd/ReadVariableOp2r
7spectral_normalization_23/dense_2/MatMul/ReadVariableOp7spectral_normalization_23/dense_2/MatMul/ReadVariableOp2~
=spectral_normalization_23/spectral_normalize/AssignVariableOp=spectral_normalization_23/spectral_normalize/AssignVariableOp2�
?spectral_normalization_23/spectral_normalize/AssignVariableOp_1?spectral_normalization_23/spectral_normalize/AssignVariableOp_12�
Bspectral_normalization_23/spectral_normalize/MatMul/ReadVariableOpBspectral_normalization_23/spectral_normalize/MatMul/ReadVariableOp2z
;spectral_normalization_23/spectral_normalize/ReadVariableOp;spectral_normalization_23/spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�2
�
I__inference_discriminator_layer_call_and_return_conditional_losses_733676
input_6:
 spectral_normalization_20_733639:@.
 spectral_normalization_20_733641:@*
layer_normalization_6_733644:@*
layer_normalization_6_733646:@:
 spectral_normalization_21_733649:	@ .
 spectral_normalization_21_733651: *
layer_normalization_7_733654: *
layer_normalization_7_733656: :
 spectral_normalization_22_733659: .
 spectral_normalization_22_733661:*
layer_normalization_8_733664:*
layer_normalization_8_733666:3
 spectral_normalization_23_733670:	�.
 spectral_normalization_23_733672:
identity��-layer_normalization_6/StatefulPartitionedCall�-layer_normalization_7/StatefulPartitionedCall�-layer_normalization_8/StatefulPartitionedCall�1spectral_normalization_20/StatefulPartitionedCall�1spectral_normalization_21/StatefulPartitionedCall�1spectral_normalization_22/StatefulPartitionedCall�1spectral_normalization_23/StatefulPartitionedCall�
1spectral_normalization_20/StatefulPartitionedCallStatefulPartitionedCallinput_6 spectral_normalization_20_733639 spectral_normalization_20_733641*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_20_layer_call_and_return_conditional_losses_732924�
-layer_normalization_6/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_20/StatefulPartitionedCall:output:0layer_normalization_6_733644layer_normalization_6_733646*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_732982�
1spectral_normalization_21/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_6/StatefulPartitionedCall:output:0 spectral_normalization_21_733649 spectral_normalization_21_733651*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������< *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_21_layer_call_and_return_conditional_losses_732999�
-layer_normalization_7/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_21/StatefulPartitionedCall:output:0layer_normalization_7_733654layer_normalization_7_733656*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������< *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_733057�
1spectral_normalization_22/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_7/StatefulPartitionedCall:output:0 spectral_normalization_22_733659 spectral_normalization_22_733661*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_22_layer_call_and_return_conditional_losses_733074�
-layer_normalization_8/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_22/StatefulPartitionedCall:output:0layer_normalization_8_733664layer_normalization_8_733666*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_layer_normalization_8_layer_call_and_return_conditional_losses_733132�
flatten_2/PartitionedCallPartitionedCall6layer_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_733144�
1spectral_normalization_23/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0 spectral_normalization_23_733670 spectral_normalization_23_733672*
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
GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_23_layer_call_and_return_conditional_losses_733156�
IdentityIdentity:spectral_normalization_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^layer_normalization_6/StatefulPartitionedCall.^layer_normalization_7/StatefulPartitionedCall.^layer_normalization_8/StatefulPartitionedCall2^spectral_normalization_20/StatefulPartitionedCall2^spectral_normalization_21/StatefulPartitionedCall2^spectral_normalization_22/StatefulPartitionedCall2^spectral_normalization_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������<: : : : : : : : : : : : : : 2^
-layer_normalization_6/StatefulPartitionedCall-layer_normalization_6/StatefulPartitionedCall2^
-layer_normalization_7/StatefulPartitionedCall-layer_normalization_7/StatefulPartitionedCall2^
-layer_normalization_8/StatefulPartitionedCall-layer_normalization_8/StatefulPartitionedCall2f
1spectral_normalization_20/StatefulPartitionedCall1spectral_normalization_20/StatefulPartitionedCall2f
1spectral_normalization_21/StatefulPartitionedCall1spectral_normalization_21/StatefulPartitionedCall2f
1spectral_normalization_22/StatefulPartitionedCall1spectral_normalization_22/StatefulPartitionedCall2f
1spectral_normalization_23/StatefulPartitionedCall1spectral_normalization_23/StatefulPartitionedCall:X T
/
_output_shapes
:���������<
!
_user_specified_name	input_6
��
�
!__inference__wrapped_model_732906
input_6j
Pdiscriminator_spectral_normalization_20_conv2d_18_conv2d_readvariableop_resource:@_
Qdiscriminator_spectral_normalization_20_conv2d_18_biasadd_readvariableop_resource:@O
Adiscriminator_layer_normalization_6_mul_4_readvariableop_resource:@M
?discriminator_layer_normalization_6_add_readvariableop_resource:@j
Pdiscriminator_spectral_normalization_21_conv2d_19_conv2d_readvariableop_resource:	@ _
Qdiscriminator_spectral_normalization_21_conv2d_19_biasadd_readvariableop_resource: O
Adiscriminator_layer_normalization_7_mul_4_readvariableop_resource: M
?discriminator_layer_normalization_7_add_readvariableop_resource: j
Pdiscriminator_spectral_normalization_22_conv2d_20_conv2d_readvariableop_resource: _
Qdiscriminator_spectral_normalization_22_conv2d_20_biasadd_readvariableop_resource:O
Adiscriminator_layer_normalization_8_mul_4_readvariableop_resource:M
?discriminator_layer_normalization_8_add_readvariableop_resource:a
Ndiscriminator_spectral_normalization_23_dense_2_matmul_readvariableop_resource:	�]
Odiscriminator_spectral_normalization_23_dense_2_biasadd_readvariableop_resource:
identity��6discriminator/layer_normalization_6/add/ReadVariableOp�8discriminator/layer_normalization_6/mul_4/ReadVariableOp�6discriminator/layer_normalization_7/add/ReadVariableOp�8discriminator/layer_normalization_7/mul_4/ReadVariableOp�6discriminator/layer_normalization_8/add/ReadVariableOp�8discriminator/layer_normalization_8/mul_4/ReadVariableOp�Hdiscriminator/spectral_normalization_20/conv2d_18/BiasAdd/ReadVariableOp�Gdiscriminator/spectral_normalization_20/conv2d_18/Conv2D/ReadVariableOp�Hdiscriminator/spectral_normalization_21/conv2d_19/BiasAdd/ReadVariableOp�Gdiscriminator/spectral_normalization_21/conv2d_19/Conv2D/ReadVariableOp�Hdiscriminator/spectral_normalization_22/conv2d_20/BiasAdd/ReadVariableOp�Gdiscriminator/spectral_normalization_22/conv2d_20/Conv2D/ReadVariableOp�Fdiscriminator/spectral_normalization_23/dense_2/BiasAdd/ReadVariableOp�Ediscriminator/spectral_normalization_23/dense_2/MatMul/ReadVariableOp�
Gdiscriminator/spectral_normalization_20/conv2d_18/Conv2D/ReadVariableOpReadVariableOpPdiscriminator_spectral_normalization_20_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
8discriminator/spectral_normalization_20/conv2d_18/Conv2DConv2Dinput_6Odiscriminator/spectral_normalization_20/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<@*
paddingSAME*
strides
�
Hdiscriminator/spectral_normalization_20/conv2d_18/BiasAdd/ReadVariableOpReadVariableOpQdiscriminator_spectral_normalization_20_conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
9discriminator/spectral_normalization_20/conv2d_18/BiasAddBiasAddAdiscriminator/spectral_normalization_20/conv2d_18/Conv2D:output:0Pdiscriminator/spectral_normalization_20/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<@�
Idiscriminator/spectral_normalization_20/conv2d_18/leaky_re_lu_2/LeakyRelu	LeakyReluBdiscriminator/spectral_normalization_20/conv2d_18/BiasAdd:output:0*/
_output_shapes
:���������<@�
)discriminator/layer_normalization_6/ShapeShapeWdiscriminator/spectral_normalization_20/conv2d_18/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:�
7discriminator/layer_normalization_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
9discriminator/layer_normalization_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
9discriminator/layer_normalization_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1discriminator/layer_normalization_6/strided_sliceStridedSlice2discriminator/layer_normalization_6/Shape:output:0@discriminator/layer_normalization_6/strided_slice/stack:output:0Bdiscriminator/layer_normalization_6/strided_slice/stack_1:output:0Bdiscriminator/layer_normalization_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)discriminator/layer_normalization_6/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
'discriminator/layer_normalization_6/mulMul2discriminator/layer_normalization_6/mul/x:output:0:discriminator/layer_normalization_6/strided_slice:output:0*
T0*
_output_shapes
: �
9discriminator/layer_normalization_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3discriminator/layer_normalization_6/strided_slice_1StridedSlice2discriminator/layer_normalization_6/Shape:output:0Bdiscriminator/layer_normalization_6/strided_slice_1/stack:output:0Ddiscriminator/layer_normalization_6/strided_slice_1/stack_1:output:0Ddiscriminator/layer_normalization_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
)discriminator/layer_normalization_6/mul_1Mul+discriminator/layer_normalization_6/mul:z:0<discriminator/layer_normalization_6/strided_slice_1:output:0*
T0*
_output_shapes
: �
9discriminator/layer_normalization_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3discriminator/layer_normalization_6/strided_slice_2StridedSlice2discriminator/layer_normalization_6/Shape:output:0Bdiscriminator/layer_normalization_6/strided_slice_2/stack:output:0Ddiscriminator/layer_normalization_6/strided_slice_2/stack_1:output:0Ddiscriminator/layer_normalization_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
)discriminator/layer_normalization_6/mul_2Mul-discriminator/layer_normalization_6/mul_1:z:0<discriminator/layer_normalization_6/strided_slice_2:output:0*
T0*
_output_shapes
: �
9discriminator/layer_normalization_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3discriminator/layer_normalization_6/strided_slice_3StridedSlice2discriminator/layer_normalization_6/Shape:output:0Bdiscriminator/layer_normalization_6/strided_slice_3/stack:output:0Ddiscriminator/layer_normalization_6/strided_slice_3/stack_1:output:0Ddiscriminator/layer_normalization_6/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+discriminator/layer_normalization_6/mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :�
)discriminator/layer_normalization_6/mul_3Mul4discriminator/layer_normalization_6/mul_3/x:output:0<discriminator/layer_normalization_6/strided_slice_3:output:0*
T0*
_output_shapes
: u
3discriminator/layer_normalization_6/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :u
3discriminator/layer_normalization_6/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
1discriminator/layer_normalization_6/Reshape/shapePack<discriminator/layer_normalization_6/Reshape/shape/0:output:0-discriminator/layer_normalization_6/mul_2:z:0-discriminator/layer_normalization_6/mul_3:z:0<discriminator/layer_normalization_6/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
+discriminator/layer_normalization_6/ReshapeReshapeWdiscriminator/spectral_normalization_20/conv2d_18/leaky_re_lu_2/LeakyRelu:activations:0:discriminator/layer_normalization_6/Reshape/shape:output:0*
T0*/
_output_shapes
:���������@�
/discriminator/layer_normalization_6/ones/packedPack-discriminator/layer_normalization_6/mul_2:z:0*
N*
T0*
_output_shapes
:s
.discriminator/layer_normalization_6/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
(discriminator/layer_normalization_6/onesFill8discriminator/layer_normalization_6/ones/packed:output:07discriminator/layer_normalization_6/ones/Const:output:0*
T0*#
_output_shapes
:����������
0discriminator/layer_normalization_6/zeros/packedPack-discriminator/layer_normalization_6/mul_2:z:0*
N*
T0*
_output_shapes
:t
/discriminator/layer_normalization_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
)discriminator/layer_normalization_6/zerosFill9discriminator/layer_normalization_6/zeros/packed:output:08discriminator/layer_normalization_6/zeros/Const:output:0*
T0*#
_output_shapes
:���������l
)discriminator/layer_normalization_6/ConstConst*
_output_shapes
: *
dtype0*
valueB n
+discriminator/layer_normalization_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
4discriminator/layer_normalization_6/FusedBatchNormV3FusedBatchNormV34discriminator/layer_normalization_6/Reshape:output:01discriminator/layer_normalization_6/ones:output:02discriminator/layer_normalization_6/zeros:output:02discriminator/layer_normalization_6/Const:output:04discriminator/layer_normalization_6/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������@:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
-discriminator/layer_normalization_6/Reshape_1Reshape8discriminator/layer_normalization_6/FusedBatchNormV3:y:02discriminator/layer_normalization_6/Shape:output:0*
T0*/
_output_shapes
:���������<@�
8discriminator/layer_normalization_6/mul_4/ReadVariableOpReadVariableOpAdiscriminator_layer_normalization_6_mul_4_readvariableop_resource*
_output_shapes
:@*
dtype0�
)discriminator/layer_normalization_6/mul_4Mul6discriminator/layer_normalization_6/Reshape_1:output:0@discriminator/layer_normalization_6/mul_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<@�
6discriminator/layer_normalization_6/add/ReadVariableOpReadVariableOp?discriminator_layer_normalization_6_add_readvariableop_resource*
_output_shapes
:@*
dtype0�
'discriminator/layer_normalization_6/addAddV2-discriminator/layer_normalization_6/mul_4:z:0>discriminator/layer_normalization_6/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<@�
Gdiscriminator/spectral_normalization_21/conv2d_19/Conv2D/ReadVariableOpReadVariableOpPdiscriminator_spectral_normalization_21_conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:	@ *
dtype0�
8discriminator/spectral_normalization_21/conv2d_19/Conv2DConv2D+discriminator/layer_normalization_6/add:z:0Odiscriminator/spectral_normalization_21/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< *
paddingSAME*
strides
�
Hdiscriminator/spectral_normalization_21/conv2d_19/BiasAdd/ReadVariableOpReadVariableOpQdiscriminator_spectral_normalization_21_conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
9discriminator/spectral_normalization_21/conv2d_19/BiasAddBiasAddAdiscriminator/spectral_normalization_21/conv2d_19/Conv2D:output:0Pdiscriminator/spectral_normalization_21/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< �
Idiscriminator/spectral_normalization_21/conv2d_19/leaky_re_lu_2/LeakyRelu	LeakyReluBdiscriminator/spectral_normalization_21/conv2d_19/BiasAdd:output:0*/
_output_shapes
:���������< �
)discriminator/layer_normalization_7/ShapeShapeWdiscriminator/spectral_normalization_21/conv2d_19/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:�
7discriminator/layer_normalization_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
9discriminator/layer_normalization_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
9discriminator/layer_normalization_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1discriminator/layer_normalization_7/strided_sliceStridedSlice2discriminator/layer_normalization_7/Shape:output:0@discriminator/layer_normalization_7/strided_slice/stack:output:0Bdiscriminator/layer_normalization_7/strided_slice/stack_1:output:0Bdiscriminator/layer_normalization_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)discriminator/layer_normalization_7/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
'discriminator/layer_normalization_7/mulMul2discriminator/layer_normalization_7/mul/x:output:0:discriminator/layer_normalization_7/strided_slice:output:0*
T0*
_output_shapes
: �
9discriminator/layer_normalization_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3discriminator/layer_normalization_7/strided_slice_1StridedSlice2discriminator/layer_normalization_7/Shape:output:0Bdiscriminator/layer_normalization_7/strided_slice_1/stack:output:0Ddiscriminator/layer_normalization_7/strided_slice_1/stack_1:output:0Ddiscriminator/layer_normalization_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
)discriminator/layer_normalization_7/mul_1Mul+discriminator/layer_normalization_7/mul:z:0<discriminator/layer_normalization_7/strided_slice_1:output:0*
T0*
_output_shapes
: �
9discriminator/layer_normalization_7/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_7/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_7/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3discriminator/layer_normalization_7/strided_slice_2StridedSlice2discriminator/layer_normalization_7/Shape:output:0Bdiscriminator/layer_normalization_7/strided_slice_2/stack:output:0Ddiscriminator/layer_normalization_7/strided_slice_2/stack_1:output:0Ddiscriminator/layer_normalization_7/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
)discriminator/layer_normalization_7/mul_2Mul-discriminator/layer_normalization_7/mul_1:z:0<discriminator/layer_normalization_7/strided_slice_2:output:0*
T0*
_output_shapes
: �
9discriminator/layer_normalization_7/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_7/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_7/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3discriminator/layer_normalization_7/strided_slice_3StridedSlice2discriminator/layer_normalization_7/Shape:output:0Bdiscriminator/layer_normalization_7/strided_slice_3/stack:output:0Ddiscriminator/layer_normalization_7/strided_slice_3/stack_1:output:0Ddiscriminator/layer_normalization_7/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+discriminator/layer_normalization_7/mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :�
)discriminator/layer_normalization_7/mul_3Mul4discriminator/layer_normalization_7/mul_3/x:output:0<discriminator/layer_normalization_7/strided_slice_3:output:0*
T0*
_output_shapes
: u
3discriminator/layer_normalization_7/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :u
3discriminator/layer_normalization_7/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
1discriminator/layer_normalization_7/Reshape/shapePack<discriminator/layer_normalization_7/Reshape/shape/0:output:0-discriminator/layer_normalization_7/mul_2:z:0-discriminator/layer_normalization_7/mul_3:z:0<discriminator/layer_normalization_7/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
+discriminator/layer_normalization_7/ReshapeReshapeWdiscriminator/spectral_normalization_21/conv2d_19/leaky_re_lu_2/LeakyRelu:activations:0:discriminator/layer_normalization_7/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� �
/discriminator/layer_normalization_7/ones/packedPack-discriminator/layer_normalization_7/mul_2:z:0*
N*
T0*
_output_shapes
:s
.discriminator/layer_normalization_7/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
(discriminator/layer_normalization_7/onesFill8discriminator/layer_normalization_7/ones/packed:output:07discriminator/layer_normalization_7/ones/Const:output:0*
T0*#
_output_shapes
:����������
0discriminator/layer_normalization_7/zeros/packedPack-discriminator/layer_normalization_7/mul_2:z:0*
N*
T0*
_output_shapes
:t
/discriminator/layer_normalization_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
)discriminator/layer_normalization_7/zerosFill9discriminator/layer_normalization_7/zeros/packed:output:08discriminator/layer_normalization_7/zeros/Const:output:0*
T0*#
_output_shapes
:���������l
)discriminator/layer_normalization_7/ConstConst*
_output_shapes
: *
dtype0*
valueB n
+discriminator/layer_normalization_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
4discriminator/layer_normalization_7/FusedBatchNormV3FusedBatchNormV34discriminator/layer_normalization_7/Reshape:output:01discriminator/layer_normalization_7/ones:output:02discriminator/layer_normalization_7/zeros:output:02discriminator/layer_normalization_7/Const:output:04discriminator/layer_normalization_7/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:��������� :���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
-discriminator/layer_normalization_7/Reshape_1Reshape8discriminator/layer_normalization_7/FusedBatchNormV3:y:02discriminator/layer_normalization_7/Shape:output:0*
T0*/
_output_shapes
:���������< �
8discriminator/layer_normalization_7/mul_4/ReadVariableOpReadVariableOpAdiscriminator_layer_normalization_7_mul_4_readvariableop_resource*
_output_shapes
: *
dtype0�
)discriminator/layer_normalization_7/mul_4Mul6discriminator/layer_normalization_7/Reshape_1:output:0@discriminator/layer_normalization_7/mul_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< �
6discriminator/layer_normalization_7/add/ReadVariableOpReadVariableOp?discriminator_layer_normalization_7_add_readvariableop_resource*
_output_shapes
: *
dtype0�
'discriminator/layer_normalization_7/addAddV2-discriminator/layer_normalization_7/mul_4:z:0>discriminator/layer_normalization_7/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< �
Gdiscriminator/spectral_normalization_22/conv2d_20/Conv2D/ReadVariableOpReadVariableOpPdiscriminator_spectral_normalization_22_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
8discriminator/spectral_normalization_22/conv2d_20/Conv2DConv2D+discriminator/layer_normalization_7/add:z:0Odiscriminator/spectral_normalization_22/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<*
paddingSAME*
strides
�
Hdiscriminator/spectral_normalization_22/conv2d_20/BiasAdd/ReadVariableOpReadVariableOpQdiscriminator_spectral_normalization_22_conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
9discriminator/spectral_normalization_22/conv2d_20/BiasAddBiasAddAdiscriminator/spectral_normalization_22/conv2d_20/Conv2D:output:0Pdiscriminator/spectral_normalization_22/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<�
Idiscriminator/spectral_normalization_22/conv2d_20/leaky_re_lu_2/LeakyRelu	LeakyReluBdiscriminator/spectral_normalization_22/conv2d_20/BiasAdd:output:0*/
_output_shapes
:���������<�
)discriminator/layer_normalization_8/ShapeShapeWdiscriminator/spectral_normalization_22/conv2d_20/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:�
7discriminator/layer_normalization_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
9discriminator/layer_normalization_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
9discriminator/layer_normalization_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1discriminator/layer_normalization_8/strided_sliceStridedSlice2discriminator/layer_normalization_8/Shape:output:0@discriminator/layer_normalization_8/strided_slice/stack:output:0Bdiscriminator/layer_normalization_8/strided_slice/stack_1:output:0Bdiscriminator/layer_normalization_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)discriminator/layer_normalization_8/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
'discriminator/layer_normalization_8/mulMul2discriminator/layer_normalization_8/mul/x:output:0:discriminator/layer_normalization_8/strided_slice:output:0*
T0*
_output_shapes
: �
9discriminator/layer_normalization_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3discriminator/layer_normalization_8/strided_slice_1StridedSlice2discriminator/layer_normalization_8/Shape:output:0Bdiscriminator/layer_normalization_8/strided_slice_1/stack:output:0Ddiscriminator/layer_normalization_8/strided_slice_1/stack_1:output:0Ddiscriminator/layer_normalization_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
)discriminator/layer_normalization_8/mul_1Mul+discriminator/layer_normalization_8/mul:z:0<discriminator/layer_normalization_8/strided_slice_1:output:0*
T0*
_output_shapes
: �
9discriminator/layer_normalization_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3discriminator/layer_normalization_8/strided_slice_2StridedSlice2discriminator/layer_normalization_8/Shape:output:0Bdiscriminator/layer_normalization_8/strided_slice_2/stack:output:0Ddiscriminator/layer_normalization_8/strided_slice_2/stack_1:output:0Ddiscriminator/layer_normalization_8/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
)discriminator/layer_normalization_8/mul_2Mul-discriminator/layer_normalization_8/mul_1:z:0<discriminator/layer_normalization_8/strided_slice_2:output:0*
T0*
_output_shapes
: �
9discriminator/layer_normalization_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3discriminator/layer_normalization_8/strided_slice_3StridedSlice2discriminator/layer_normalization_8/Shape:output:0Bdiscriminator/layer_normalization_8/strided_slice_3/stack:output:0Ddiscriminator/layer_normalization_8/strided_slice_3/stack_1:output:0Ddiscriminator/layer_normalization_8/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+discriminator/layer_normalization_8/mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :�
)discriminator/layer_normalization_8/mul_3Mul4discriminator/layer_normalization_8/mul_3/x:output:0<discriminator/layer_normalization_8/strided_slice_3:output:0*
T0*
_output_shapes
: u
3discriminator/layer_normalization_8/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :u
3discriminator/layer_normalization_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
1discriminator/layer_normalization_8/Reshape/shapePack<discriminator/layer_normalization_8/Reshape/shape/0:output:0-discriminator/layer_normalization_8/mul_2:z:0-discriminator/layer_normalization_8/mul_3:z:0<discriminator/layer_normalization_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
+discriminator/layer_normalization_8/ReshapeReshapeWdiscriminator/spectral_normalization_22/conv2d_20/leaky_re_lu_2/LeakyRelu:activations:0:discriminator/layer_normalization_8/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
/discriminator/layer_normalization_8/ones/packedPack-discriminator/layer_normalization_8/mul_2:z:0*
N*
T0*
_output_shapes
:s
.discriminator/layer_normalization_8/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
(discriminator/layer_normalization_8/onesFill8discriminator/layer_normalization_8/ones/packed:output:07discriminator/layer_normalization_8/ones/Const:output:0*
T0*#
_output_shapes
:����������
0discriminator/layer_normalization_8/zeros/packedPack-discriminator/layer_normalization_8/mul_2:z:0*
N*
T0*
_output_shapes
:t
/discriminator/layer_normalization_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
)discriminator/layer_normalization_8/zerosFill9discriminator/layer_normalization_8/zeros/packed:output:08discriminator/layer_normalization_8/zeros/Const:output:0*
T0*#
_output_shapes
:���������l
)discriminator/layer_normalization_8/ConstConst*
_output_shapes
: *
dtype0*
valueB n
+discriminator/layer_normalization_8/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
4discriminator/layer_normalization_8/FusedBatchNormV3FusedBatchNormV34discriminator/layer_normalization_8/Reshape:output:01discriminator/layer_normalization_8/ones:output:02discriminator/layer_normalization_8/zeros:output:02discriminator/layer_normalization_8/Const:output:04discriminator/layer_normalization_8/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
-discriminator/layer_normalization_8/Reshape_1Reshape8discriminator/layer_normalization_8/FusedBatchNormV3:y:02discriminator/layer_normalization_8/Shape:output:0*
T0*/
_output_shapes
:���������<�
8discriminator/layer_normalization_8/mul_4/ReadVariableOpReadVariableOpAdiscriminator_layer_normalization_8_mul_4_readvariableop_resource*
_output_shapes
:*
dtype0�
)discriminator/layer_normalization_8/mul_4Mul6discriminator/layer_normalization_8/Reshape_1:output:0@discriminator/layer_normalization_8/mul_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<�
6discriminator/layer_normalization_8/add/ReadVariableOpReadVariableOp?discriminator_layer_normalization_8_add_readvariableop_resource*
_output_shapes
:*
dtype0�
'discriminator/layer_normalization_8/addAddV2-discriminator/layer_normalization_8/mul_4:z:0>discriminator/layer_normalization_8/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<n
discriminator/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  �
discriminator/flatten_2/ReshapeReshape+discriminator/layer_normalization_8/add:z:0&discriminator/flatten_2/Const:output:0*
T0*(
_output_shapes
:�����������
Ediscriminator/spectral_normalization_23/dense_2/MatMul/ReadVariableOpReadVariableOpNdiscriminator_spectral_normalization_23_dense_2_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
6discriminator/spectral_normalization_23/dense_2/MatMulMatMul(discriminator/flatten_2/Reshape:output:0Mdiscriminator/spectral_normalization_23/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Fdiscriminator/spectral_normalization_23/dense_2/BiasAdd/ReadVariableOpReadVariableOpOdiscriminator_spectral_normalization_23_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
7discriminator/spectral_normalization_23/dense_2/BiasAddBiasAdd@discriminator/spectral_normalization_23/dense_2/MatMul:product:0Ndiscriminator/spectral_normalization_23/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
IdentityIdentity@discriminator/spectral_normalization_23/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp7^discriminator/layer_normalization_6/add/ReadVariableOp9^discriminator/layer_normalization_6/mul_4/ReadVariableOp7^discriminator/layer_normalization_7/add/ReadVariableOp9^discriminator/layer_normalization_7/mul_4/ReadVariableOp7^discriminator/layer_normalization_8/add/ReadVariableOp9^discriminator/layer_normalization_8/mul_4/ReadVariableOpI^discriminator/spectral_normalization_20/conv2d_18/BiasAdd/ReadVariableOpH^discriminator/spectral_normalization_20/conv2d_18/Conv2D/ReadVariableOpI^discriminator/spectral_normalization_21/conv2d_19/BiasAdd/ReadVariableOpH^discriminator/spectral_normalization_21/conv2d_19/Conv2D/ReadVariableOpI^discriminator/spectral_normalization_22/conv2d_20/BiasAdd/ReadVariableOpH^discriminator/spectral_normalization_22/conv2d_20/Conv2D/ReadVariableOpG^discriminator/spectral_normalization_23/dense_2/BiasAdd/ReadVariableOpF^discriminator/spectral_normalization_23/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������<: : : : : : : : : : : : : : 2p
6discriminator/layer_normalization_6/add/ReadVariableOp6discriminator/layer_normalization_6/add/ReadVariableOp2t
8discriminator/layer_normalization_6/mul_4/ReadVariableOp8discriminator/layer_normalization_6/mul_4/ReadVariableOp2p
6discriminator/layer_normalization_7/add/ReadVariableOp6discriminator/layer_normalization_7/add/ReadVariableOp2t
8discriminator/layer_normalization_7/mul_4/ReadVariableOp8discriminator/layer_normalization_7/mul_4/ReadVariableOp2p
6discriminator/layer_normalization_8/add/ReadVariableOp6discriminator/layer_normalization_8/add/ReadVariableOp2t
8discriminator/layer_normalization_8/mul_4/ReadVariableOp8discriminator/layer_normalization_8/mul_4/ReadVariableOp2�
Hdiscriminator/spectral_normalization_20/conv2d_18/BiasAdd/ReadVariableOpHdiscriminator/spectral_normalization_20/conv2d_18/BiasAdd/ReadVariableOp2�
Gdiscriminator/spectral_normalization_20/conv2d_18/Conv2D/ReadVariableOpGdiscriminator/spectral_normalization_20/conv2d_18/Conv2D/ReadVariableOp2�
Hdiscriminator/spectral_normalization_21/conv2d_19/BiasAdd/ReadVariableOpHdiscriminator/spectral_normalization_21/conv2d_19/BiasAdd/ReadVariableOp2�
Gdiscriminator/spectral_normalization_21/conv2d_19/Conv2D/ReadVariableOpGdiscriminator/spectral_normalization_21/conv2d_19/Conv2D/ReadVariableOp2�
Hdiscriminator/spectral_normalization_22/conv2d_20/BiasAdd/ReadVariableOpHdiscriminator/spectral_normalization_22/conv2d_20/BiasAdd/ReadVariableOp2�
Gdiscriminator/spectral_normalization_22/conv2d_20/Conv2D/ReadVariableOpGdiscriminator/spectral_normalization_22/conv2d_20/Conv2D/ReadVariableOp2�
Fdiscriminator/spectral_normalization_23/dense_2/BiasAdd/ReadVariableOpFdiscriminator/spectral_normalization_23/dense_2/BiasAdd/ReadVariableOp2�
Ediscriminator/spectral_normalization_23/dense_2/MatMul/ReadVariableOpEdiscriminator/spectral_normalization_23/dense_2/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������<
!
_user_specified_name	input_6
�
�
6__inference_layer_normalization_7_layer_call_fn_734515

inputs
unknown: 
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
:���������< *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_733057w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������< `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������< : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������< 
 
_user_specified_nameinputs
�5
�
U__inference_spectral_normalization_23_layer_call_and_return_conditional_losses_734779

inputs2
reshape_readvariableop_resource:	�C
1spectral_normalize_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identity��Reshape/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOpw
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:	�*
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   t
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	��
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes
:	�*
transpose_b(
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	�v
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
:	��
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes

:�
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:x
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

:
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:�
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	��
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes

:�
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
!spectral_normalize/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:	�*
dtype0�
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*
_output_shapes
:	�q
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�     �
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*
_output_shapes
:	��
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
dense_2/MatMul/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*
_output_shapes
:	�*
dtype0y
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^Reshape/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:����������: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_68
serving_default_input_6:0���������<M
spectral_normalization_230
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer-7
	layer_with_weights-6
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	layer
w
w_shape
sn_u
u"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"axis
	#gamma
$beta"
_tf_keras_layer
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
	+layer
,w
-w_shape
.sn_u
.u"
_tf_keras_layer
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
5axis
	6gamma
7beta"
_tf_keras_layer
�
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses
	>layer
?w
@w_shape
Asn_u
Au"
_tf_keras_layer
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
Haxis
	Igamma
Jbeta"
_tf_keras_layer
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses
	Wlayer
Xw
Yw_shape
Zsn_u
Zu"
_tf_keras_layer
�
0
[1
2
#3
$4
,5
\6
.7
68
79
?10
]11
A12
I13
J14
X15
^16
Z17"
trackable_list_wrapper
�
0
[1
#2
$3
,4
\5
66
77
?8
]9
I10
J11
X12
^13"
trackable_list_wrapper
 "
trackable_list_wrapper
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
dtrace_0
etrace_1
ftrace_2
gtrace_32�
.__inference_discriminator_layer_call_fn_733194
.__inference_discriminator_layer_call_fn_733792
.__inference_discriminator_layer_call_fn_733833
.__inference_discriminator_layer_call_fn_733636�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zdtrace_0zetrace_1zftrace_2zgtrace_3
�
htrace_0
itrace_1
jtrace_2
ktrace_32�
I__inference_discriminator_layer_call_and_return_conditional_losses_734010
I__inference_discriminator_layer_call_and_return_conditional_losses_734303
I__inference_discriminator_layer_call_and_return_conditional_losses_733676
I__inference_discriminator_layer_call_and_return_conditional_losses_733724�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zhtrace_0zitrace_1zjtrace_2zktrace_3
�B�
!__inference__wrapped_model_732906input_6"�
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
lserving_default"
signature_map
5
0
[1
2"
trackable_list_wrapper
.
0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
rtrace_0
strace_12�
:__inference_spectral_normalization_20_layer_call_fn_734312
:__inference_spectral_normalization_20_layer_call_fn_734323�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zrtrace_0zstrace_1
�
ttrace_0
utrace_12�
U__inference_spectral_normalization_20_layer_call_and_return_conditional_losses_734334
U__inference_spectral_normalization_20_layer_call_and_return_conditional_losses_734374�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zttrace_0zutrace_1
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses
|
activation

kernel
[bias
 }_jit_compiled_convolution_op"
_tf_keras_layer
::8@2 spectral_normalization_20/kernel
 "
trackable_list_wrapper
.:,@2spectral_normalization_20/sn_u
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
~non_trainable_variables

layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
6__inference_layer_normalization_6_layer_call_fn_734383�
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
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_734435�
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
 "
trackable_list_wrapper
):'@2layer_normalization_6/gamma
(:&@2layer_normalization_6/beta
5
,0
\1
.2"
trackable_list_wrapper
.
,0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
:__inference_spectral_normalization_21_layer_call_fn_734444
:__inference_spectral_normalization_21_layer_call_fn_734455�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
U__inference_spectral_normalization_21_layer_call_and_return_conditional_losses_734466
U__inference_spectral_normalization_21_layer_call_and_return_conditional_losses_734506�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
|
activation

,kernel
\bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
::8	@ 2 spectral_normalization_21/kernel
 "
trackable_list_wrapper
.:, 2spectral_normalization_21/sn_u
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
6__inference_layer_normalization_7_layer_call_fn_734515�
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
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_734567�
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
 "
trackable_list_wrapper
):' 2layer_normalization_7/gamma
(:& 2layer_normalization_7/beta
5
?0
]1
A2"
trackable_list_wrapper
.
?0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
:__inference_spectral_normalization_22_layer_call_fn_734576
:__inference_spectral_normalization_22_layer_call_fn_734587�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
U__inference_spectral_normalization_22_layer_call_and_return_conditional_losses_734598
U__inference_spectral_normalization_22_layer_call_and_return_conditional_losses_734638�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
|
activation

?kernel
]bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
::8 2 spectral_normalization_22/kernel
 "
trackable_list_wrapper
.:,2spectral_normalization_22/sn_u
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
6__inference_layer_normalization_8_layer_call_fn_734647�
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
Q__inference_layer_normalization_8_layer_call_and_return_conditional_losses_734699�
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
 "
trackable_list_wrapper
):'2layer_normalization_8/gamma
(:&2layer_normalization_8/beta
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
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_flatten_2_layer_call_fn_734704�
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
E__inference_flatten_2_layer_call_and_return_conditional_losses_734710�
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
5
X0
^1
Z2"
trackable_list_wrapper
.
X0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
:__inference_spectral_normalization_23_layer_call_fn_734719
:__inference_spectral_normalization_23_layer_call_fn_734730�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
U__inference_spectral_normalization_23_layer_call_and_return_conditional_losses_734740
U__inference_spectral_normalization_23_layer_call_and_return_conditional_losses_734779�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Xkernel
^bias"
_tf_keras_layer
3:1	�2 spectral_normalization_23/kernel
 "
trackable_list_wrapper
.:,2spectral_normalization_23/sn_u
,:*@2spectral_normalization_20/bias
,:* 2spectral_normalization_21/bias
,:*2spectral_normalization_22/bias
,:*2spectral_normalization_23/bias
<
0
.1
A2
Z3"
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_discriminator_layer_call_fn_733194input_6"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_discriminator_layer_call_fn_733792inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_discriminator_layer_call_fn_733833inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_discriminator_layer_call_fn_733636input_6"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_discriminator_layer_call_and_return_conditional_losses_734010inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_discriminator_layer_call_and_return_conditional_losses_734303inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_discriminator_layer_call_and_return_conditional_losses_733676input_6"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_discriminator_layer_call_and_return_conditional_losses_733724input_6"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_signature_wrapper_733759input_6"�
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
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
:__inference_spectral_normalization_20_layer_call_fn_734312inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
:__inference_spectral_normalization_20_layer_call_fn_734323inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
U__inference_spectral_normalization_20_layer_call_and_return_conditional_losses_734334inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
U__inference_spectral_normalization_20_layer_call_and_return_conditional_losses_734374inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
[1"
trackable_list_wrapper
.
0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
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
6__inference_layer_normalization_6_layer_call_fn_734383inputs"�
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
�B�
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_734435inputs"�
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
'
.0"
trackable_list_wrapper
'
+0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
:__inference_spectral_normalization_21_layer_call_fn_734444inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
:__inference_spectral_normalization_21_layer_call_fn_734455inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
U__inference_spectral_normalization_21_layer_call_and_return_conditional_losses_734466inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
U__inference_spectral_normalization_21_layer_call_and_return_conditional_losses_734506inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
,0
\1"
trackable_list_wrapper
.
,0
\1"
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
6__inference_layer_normalization_7_layer_call_fn_734515inputs"�
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
�B�
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_734567inputs"�
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
'
A0"
trackable_list_wrapper
'
>0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
:__inference_spectral_normalization_22_layer_call_fn_734576inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
:__inference_spectral_normalization_22_layer_call_fn_734587inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
U__inference_spectral_normalization_22_layer_call_and_return_conditional_losses_734598inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
U__inference_spectral_normalization_22_layer_call_and_return_conditional_losses_734638inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
?0
]1"
trackable_list_wrapper
.
?0
]1"
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
6__inference_layer_normalization_8_layer_call_fn_734647inputs"�
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
�B�
Q__inference_layer_normalization_8_layer_call_and_return_conditional_losses_734699inputs"�
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
*__inference_flatten_2_layer_call_fn_734704inputs"�
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
E__inference_flatten_2_layer_call_and_return_conditional_losses_734710inputs"�
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
'
Z0"
trackable_list_wrapper
'
W0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
:__inference_spectral_normalization_23_layer_call_fn_734719inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
:__inference_spectral_normalization_23_layer_call_fn_734730inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
U__inference_spectral_normalization_23_layer_call_and_return_conditional_losses_734740inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
U__inference_spectral_normalization_23_layer_call_and_return_conditional_losses_734779inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
X0
^1"
trackable_list_wrapper
.
X0
^1"
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
|0"
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
|0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
|0"
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
!__inference__wrapped_model_732906�[#$,\67?]IJX^8�5
.�+
)�&
input_6���������<
� "U�R
P
spectral_normalization_233�0
spectral_normalization_23����������
I__inference_discriminator_layer_call_and_return_conditional_losses_733676y[#$,\67?]IJX^@�=
6�3
)�&
input_6���������<
p 

 
� "%�"
�
0���������
� �
I__inference_discriminator_layer_call_and_return_conditional_losses_733724}[#$,.\67?A]IJXZ^@�=
6�3
)�&
input_6���������<
p

 
� "%�"
�
0���������
� �
I__inference_discriminator_layer_call_and_return_conditional_losses_734010x[#$,\67?]IJX^?�<
5�2
(�%
inputs���������<
p 

 
� "%�"
�
0���������
� �
I__inference_discriminator_layer_call_and_return_conditional_losses_734303|[#$,.\67?A]IJXZ^?�<
5�2
(�%
inputs���������<
p

 
� "%�"
�
0���������
� �
.__inference_discriminator_layer_call_fn_733194l[#$,\67?]IJX^@�=
6�3
)�&
input_6���������<
p 

 
� "�����������
.__inference_discriminator_layer_call_fn_733636p[#$,.\67?A]IJXZ^@�=
6�3
)�&
input_6���������<
p

 
� "�����������
.__inference_discriminator_layer_call_fn_733792k[#$,\67?]IJX^?�<
5�2
(�%
inputs���������<
p 

 
� "�����������
.__inference_discriminator_layer_call_fn_733833o[#$,.\67?A]IJXZ^?�<
5�2
(�%
inputs���������<
p

 
� "�����������
E__inference_flatten_2_layer_call_and_return_conditional_losses_734710a7�4
-�*
(�%
inputs���������<
� "&�#
�
0����������
� �
*__inference_flatten_2_layer_call_fn_734704T7�4
-�*
(�%
inputs���������<
� "������������
Q__inference_layer_normalization_6_layer_call_and_return_conditional_losses_734435l#$7�4
-�*
(�%
inputs���������<@
� "-�*
#� 
0���������<@
� �
6__inference_layer_normalization_6_layer_call_fn_734383_#$7�4
-�*
(�%
inputs���������<@
� " ����������<@�
Q__inference_layer_normalization_7_layer_call_and_return_conditional_losses_734567l677�4
-�*
(�%
inputs���������< 
� "-�*
#� 
0���������< 
� �
6__inference_layer_normalization_7_layer_call_fn_734515_677�4
-�*
(�%
inputs���������< 
� " ����������< �
Q__inference_layer_normalization_8_layer_call_and_return_conditional_losses_734699lIJ7�4
-�*
(�%
inputs���������<
� "-�*
#� 
0���������<
� �
6__inference_layer_normalization_8_layer_call_fn_734647_IJ7�4
-�*
(�%
inputs���������<
� " ����������<�
$__inference_signature_wrapper_733759�[#$,\67?]IJX^C�@
� 
9�6
4
input_6)�&
input_6���������<"U�R
P
spectral_normalization_233�0
spectral_normalization_23����������
U__inference_spectral_normalization_20_layer_call_and_return_conditional_losses_734334p[;�8
1�.
(�%
inputs���������<
p 
� "-�*
#� 
0���������<@
� �
U__inference_spectral_normalization_20_layer_call_and_return_conditional_losses_734374q[;�8
1�.
(�%
inputs���������<
p
� "-�*
#� 
0���������<@
� �
:__inference_spectral_normalization_20_layer_call_fn_734312c[;�8
1�.
(�%
inputs���������<
p 
� " ����������<@�
:__inference_spectral_normalization_20_layer_call_fn_734323d[;�8
1�.
(�%
inputs���������<
p
� " ����������<@�
U__inference_spectral_normalization_21_layer_call_and_return_conditional_losses_734466p,\;�8
1�.
(�%
inputs���������<@
p 
� "-�*
#� 
0���������< 
� �
U__inference_spectral_normalization_21_layer_call_and_return_conditional_losses_734506q,.\;�8
1�.
(�%
inputs���������<@
p
� "-�*
#� 
0���������< 
� �
:__inference_spectral_normalization_21_layer_call_fn_734444c,\;�8
1�.
(�%
inputs���������<@
p 
� " ����������< �
:__inference_spectral_normalization_21_layer_call_fn_734455d,.\;�8
1�.
(�%
inputs���������<@
p
� " ����������< �
U__inference_spectral_normalization_22_layer_call_and_return_conditional_losses_734598p?];�8
1�.
(�%
inputs���������< 
p 
� "-�*
#� 
0���������<
� �
U__inference_spectral_normalization_22_layer_call_and_return_conditional_losses_734638q?A];�8
1�.
(�%
inputs���������< 
p
� "-�*
#� 
0���������<
� �
:__inference_spectral_normalization_22_layer_call_fn_734576c?];�8
1�.
(�%
inputs���������< 
p 
� " ����������<�
:__inference_spectral_normalization_22_layer_call_fn_734587d?A];�8
1�.
(�%
inputs���������< 
p
� " ����������<�
U__inference_spectral_normalization_23_layer_call_and_return_conditional_losses_734740aX^4�1
*�'
!�
inputs����������
p 
� "%�"
�
0���������
� �
U__inference_spectral_normalization_23_layer_call_and_return_conditional_losses_734779bXZ^4�1
*�'
!�
inputs����������
p
� "%�"
�
0���������
� �
:__inference_spectral_normalization_23_layer_call_fn_734719TX^4�1
*�'
!�
inputs����������
p 
� "�����������
:__inference_spectral_normalization_23_layer_call_fn_734730UXZ^4�1
*�'
!�
inputs����������
p
� "����������
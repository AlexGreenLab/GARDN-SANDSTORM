Ə
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
spectral_normalization_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namespectral_normalization_7/bias
�
1spectral_normalization_7/bias/Read/ReadVariableOpReadVariableOpspectral_normalization_7/bias*
_output_shapes
:*
dtype0
�
spectral_normalization_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namespectral_normalization_6/bias
�
1spectral_normalization_6/bias/Read/ReadVariableOpReadVariableOpspectral_normalization_6/bias*
_output_shapes
:*
dtype0
�
spectral_normalization_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namespectral_normalization_5/bias
�
1spectral_normalization_5/bias/Read/ReadVariableOpReadVariableOpspectral_normalization_5/bias*
_output_shapes
:*
dtype0
�
spectral_normalization_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namespectral_normalization_4/bias
�
1spectral_normalization_4/bias/Read/ReadVariableOpReadVariableOpspectral_normalization_4/bias*
_output_shapes
: *
dtype0
�
spectral_normalization_7/sn_uVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_namespectral_normalization_7/sn_u
�
1spectral_normalization_7/sn_u/Read/ReadVariableOpReadVariableOpspectral_normalization_7/sn_u*
_output_shapes

:*
dtype0
�
spectral_normalization_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*0
shared_name!spectral_normalization_7/kernel
�
3spectral_normalization_7/kernel/Read/ReadVariableOpReadVariableOpspectral_normalization_7/kernel*
_output_shapes
:	�*
dtype0
�
layer_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_5/beta
�
.layer_normalization_5/beta/Read/ReadVariableOpReadVariableOplayer_normalization_5/beta*
_output_shapes
:*
dtype0
�
layer_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_5/gamma
�
/layer_normalization_5/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_5/gamma*
_output_shapes
:*
dtype0
�
spectral_normalization_6/sn_uVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_namespectral_normalization_6/sn_u
�
1spectral_normalization_6/sn_u/Read/ReadVariableOpReadVariableOpspectral_normalization_6/sn_u*
_output_shapes

:*
dtype0
�
spectral_normalization_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!spectral_normalization_6/kernel
�
3spectral_normalization_6/kernel/Read/ReadVariableOpReadVariableOpspectral_normalization_6/kernel*&
_output_shapes
:*
dtype0
�
layer_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_4/beta
�
.layer_normalization_4/beta/Read/ReadVariableOpReadVariableOplayer_normalization_4/beta*
_output_shapes
:*
dtype0
�
layer_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_4/gamma
�
/layer_normalization_4/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_4/gamma*
_output_shapes
:*
dtype0
�
spectral_normalization_5/sn_uVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_namespectral_normalization_5/sn_u
�
1spectral_normalization_5/sn_u/Read/ReadVariableOpReadVariableOpspectral_normalization_5/sn_u*
_output_shapes

:*
dtype0
�
spectral_normalization_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *0
shared_name!spectral_normalization_5/kernel
�
3spectral_normalization_5/kernel/Read/ReadVariableOpReadVariableOpspectral_normalization_5/kernel*&
_output_shapes
:	 *
dtype0
�
layer_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namelayer_normalization_3/beta
�
.layer_normalization_3/beta/Read/ReadVariableOpReadVariableOplayer_normalization_3/beta*
_output_shapes
: *
dtype0
�
layer_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namelayer_normalization_3/gamma
�
/layer_normalization_3/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_3/gamma*
_output_shapes
: *
dtype0
�
spectral_normalization_4/sn_uVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *.
shared_namespectral_normalization_4/sn_u
�
1spectral_normalization_4/sn_u/Read/ReadVariableOpReadVariableOpspectral_normalization_4/sn_u*
_output_shapes

: *
dtype0
�
spectral_normalization_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!spectral_normalization_4/kernel
�
3spectral_normalization_4/kernel/Read/ReadVariableOpReadVariableOpspectral_normalization_4/kernel*&
_output_shapes
: *
dtype0
�
serving_default_input_2Placeholder*/
_output_shapes
:���������<*
dtype0*$
shape:���������<
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2spectral_normalization_4/kernelspectral_normalization_4/biaslayer_normalization_3/gammalayer_normalization_3/betaspectral_normalization_5/kernelspectral_normalization_5/biaslayer_normalization_4/gammalayer_normalization_4/betaspectral_normalization_6/kernelspectral_normalization_6/biaslayer_normalization_5/gammalayer_normalization_5/betaspectral_normalization_7/kernelspectral_normalization_7/bias*
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
$__inference_signature_wrapper_167889

NoOpNoOp
�U
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�T
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
jd
VARIABLE_VALUEspectral_normalization_4/kernel1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUE*
* 
ke
VARIABLE_VALUEspectral_normalization_4/sn_u4layer_with_weights-0/sn_u/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUElayer_normalization_3/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_3/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
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
jd
VARIABLE_VALUEspectral_normalization_5/kernel1layer_with_weights-2/w/.ATTRIBUTES/VARIABLE_VALUE*
* 
ke
VARIABLE_VALUEspectral_normalization_5/sn_u4layer_with_weights-2/sn_u/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUElayer_normalization_4/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_4/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
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
jd
VARIABLE_VALUEspectral_normalization_6/kernel1layer_with_weights-4/w/.ATTRIBUTES/VARIABLE_VALUE*
* 
ke
VARIABLE_VALUEspectral_normalization_6/sn_u4layer_with_weights-4/sn_u/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUElayer_normalization_5/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_5/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
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
jd
VARIABLE_VALUEspectral_normalization_7/kernel1layer_with_weights-6/w/.ATTRIBUTES/VARIABLE_VALUE*
* 
ke
VARIABLE_VALUEspectral_normalization_7/sn_u4layer_with_weights-6/sn_u/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEspectral_normalization_4/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEspectral_normalization_5/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEspectral_normalization_6/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEspectral_normalization_7/bias'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
 
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
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename3spectral_normalization_4/kernel/Read/ReadVariableOp1spectral_normalization_4/sn_u/Read/ReadVariableOp/layer_normalization_3/gamma/Read/ReadVariableOp.layer_normalization_3/beta/Read/ReadVariableOp3spectral_normalization_5/kernel/Read/ReadVariableOp1spectral_normalization_5/sn_u/Read/ReadVariableOp/layer_normalization_4/gamma/Read/ReadVariableOp.layer_normalization_4/beta/Read/ReadVariableOp3spectral_normalization_6/kernel/Read/ReadVariableOp1spectral_normalization_6/sn_u/Read/ReadVariableOp/layer_normalization_5/gamma/Read/ReadVariableOp.layer_normalization_5/beta/Read/ReadVariableOp3spectral_normalization_7/kernel/Read/ReadVariableOp1spectral_normalization_7/sn_u/Read/ReadVariableOp1spectral_normalization_4/bias/Read/ReadVariableOp1spectral_normalization_5/bias/Read/ReadVariableOp1spectral_normalization_6/bias/Read/ReadVariableOp1spectral_normalization_7/bias/Read/ReadVariableOpConst*
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
__inference__traced_save_168986
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamespectral_normalization_4/kernelspectral_normalization_4/sn_ulayer_normalization_3/gammalayer_normalization_3/betaspectral_normalization_5/kernelspectral_normalization_5/sn_ulayer_normalization_4/gammalayer_normalization_4/betaspectral_normalization_6/kernelspectral_normalization_6/sn_ulayer_normalization_5/gammalayer_normalization_5/betaspectral_normalization_7/kernelspectral_normalization_7/sn_uspectral_normalization_4/biasspectral_normalization_5/biasspectral_normalization_6/biasspectral_normalization_7/bias*
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
"__inference__traced_restore_169050ؼ
�
�
6__inference_layer_normalization_3_layer_call_fn_168513

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
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_167112w
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
��
�
!__inference__wrapped_model_167036
input_2h
Ndiscriminator_spectral_normalization_4_conv2d_4_conv2d_readvariableop_resource: ]
Odiscriminator_spectral_normalization_4_conv2d_4_biasadd_readvariableop_resource: O
Adiscriminator_layer_normalization_3_mul_4_readvariableop_resource: M
?discriminator_layer_normalization_3_add_readvariableop_resource: h
Ndiscriminator_spectral_normalization_5_conv2d_5_conv2d_readvariableop_resource:	 ]
Odiscriminator_spectral_normalization_5_conv2d_5_biasadd_readvariableop_resource:O
Adiscriminator_layer_normalization_4_mul_4_readvariableop_resource:M
?discriminator_layer_normalization_4_add_readvariableop_resource:h
Ndiscriminator_spectral_normalization_6_conv2d_6_conv2d_readvariableop_resource:]
Odiscriminator_spectral_normalization_6_conv2d_6_biasadd_readvariableop_resource:O
Adiscriminator_layer_normalization_5_mul_4_readvariableop_resource:M
?discriminator_layer_normalization_5_add_readvariableop_resource:^
Kdiscriminator_spectral_normalization_7_dense_matmul_readvariableop_resource:	�Z
Ldiscriminator_spectral_normalization_7_dense_biasadd_readvariableop_resource:
identity��6discriminator/layer_normalization_3/add/ReadVariableOp�8discriminator/layer_normalization_3/mul_4/ReadVariableOp�6discriminator/layer_normalization_4/add/ReadVariableOp�8discriminator/layer_normalization_4/mul_4/ReadVariableOp�6discriminator/layer_normalization_5/add/ReadVariableOp�8discriminator/layer_normalization_5/mul_4/ReadVariableOp�Fdiscriminator/spectral_normalization_4/conv2d_4/BiasAdd/ReadVariableOp�Ediscriminator/spectral_normalization_4/conv2d_4/Conv2D/ReadVariableOp�Fdiscriminator/spectral_normalization_5/conv2d_5/BiasAdd/ReadVariableOp�Ediscriminator/spectral_normalization_5/conv2d_5/Conv2D/ReadVariableOp�Fdiscriminator/spectral_normalization_6/conv2d_6/BiasAdd/ReadVariableOp�Ediscriminator/spectral_normalization_6/conv2d_6/Conv2D/ReadVariableOp�Cdiscriminator/spectral_normalization_7/dense/BiasAdd/ReadVariableOp�Bdiscriminator/spectral_normalization_7/dense/MatMul/ReadVariableOp�
Ediscriminator/spectral_normalization_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOpNdiscriminator_spectral_normalization_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
6discriminator/spectral_normalization_4/conv2d_4/Conv2DConv2Dinput_2Mdiscriminator/spectral_normalization_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< *
paddingSAME*
strides
�
Fdiscriminator/spectral_normalization_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOpOdiscriminator_spectral_normalization_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
7discriminator/spectral_normalization_4/conv2d_4/BiasAddBiasAdd?discriminator/spectral_normalization_4/conv2d_4/Conv2D:output:0Ndiscriminator/spectral_normalization_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< �
Ediscriminator/spectral_normalization_4/conv2d_4/leaky_re_lu/LeakyRelu	LeakyRelu@discriminator/spectral_normalization_4/conv2d_4/BiasAdd:output:0*/
_output_shapes
:���������< �
)discriminator/layer_normalization_3/ShapeShapeSdiscriminator/spectral_normalization_4/conv2d_4/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:�
7discriminator/layer_normalization_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
9discriminator/layer_normalization_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
9discriminator/layer_normalization_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1discriminator/layer_normalization_3/strided_sliceStridedSlice2discriminator/layer_normalization_3/Shape:output:0@discriminator/layer_normalization_3/strided_slice/stack:output:0Bdiscriminator/layer_normalization_3/strided_slice/stack_1:output:0Bdiscriminator/layer_normalization_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)discriminator/layer_normalization_3/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
'discriminator/layer_normalization_3/mulMul2discriminator/layer_normalization_3/mul/x:output:0:discriminator/layer_normalization_3/strided_slice:output:0*
T0*
_output_shapes
: �
9discriminator/layer_normalization_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3discriminator/layer_normalization_3/strided_slice_1StridedSlice2discriminator/layer_normalization_3/Shape:output:0Bdiscriminator/layer_normalization_3/strided_slice_1/stack:output:0Ddiscriminator/layer_normalization_3/strided_slice_1/stack_1:output:0Ddiscriminator/layer_normalization_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
)discriminator/layer_normalization_3/mul_1Mul+discriminator/layer_normalization_3/mul:z:0<discriminator/layer_normalization_3/strided_slice_1:output:0*
T0*
_output_shapes
: �
9discriminator/layer_normalization_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3discriminator/layer_normalization_3/strided_slice_2StridedSlice2discriminator/layer_normalization_3/Shape:output:0Bdiscriminator/layer_normalization_3/strided_slice_2/stack:output:0Ddiscriminator/layer_normalization_3/strided_slice_2/stack_1:output:0Ddiscriminator/layer_normalization_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
)discriminator/layer_normalization_3/mul_2Mul-discriminator/layer_normalization_3/mul_1:z:0<discriminator/layer_normalization_3/strided_slice_2:output:0*
T0*
_output_shapes
: �
9discriminator/layer_normalization_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3discriminator/layer_normalization_3/strided_slice_3StridedSlice2discriminator/layer_normalization_3/Shape:output:0Bdiscriminator/layer_normalization_3/strided_slice_3/stack:output:0Ddiscriminator/layer_normalization_3/strided_slice_3/stack_1:output:0Ddiscriminator/layer_normalization_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+discriminator/layer_normalization_3/mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :�
)discriminator/layer_normalization_3/mul_3Mul4discriminator/layer_normalization_3/mul_3/x:output:0<discriminator/layer_normalization_3/strided_slice_3:output:0*
T0*
_output_shapes
: u
3discriminator/layer_normalization_3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :u
3discriminator/layer_normalization_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
1discriminator/layer_normalization_3/Reshape/shapePack<discriminator/layer_normalization_3/Reshape/shape/0:output:0-discriminator/layer_normalization_3/mul_2:z:0-discriminator/layer_normalization_3/mul_3:z:0<discriminator/layer_normalization_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
+discriminator/layer_normalization_3/ReshapeReshapeSdiscriminator/spectral_normalization_4/conv2d_4/leaky_re_lu/LeakyRelu:activations:0:discriminator/layer_normalization_3/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� �
/discriminator/layer_normalization_3/ones/packedPack-discriminator/layer_normalization_3/mul_2:z:0*
N*
T0*
_output_shapes
:s
.discriminator/layer_normalization_3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
(discriminator/layer_normalization_3/onesFill8discriminator/layer_normalization_3/ones/packed:output:07discriminator/layer_normalization_3/ones/Const:output:0*
T0*#
_output_shapes
:����������
0discriminator/layer_normalization_3/zeros/packedPack-discriminator/layer_normalization_3/mul_2:z:0*
N*
T0*
_output_shapes
:t
/discriminator/layer_normalization_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
)discriminator/layer_normalization_3/zerosFill9discriminator/layer_normalization_3/zeros/packed:output:08discriminator/layer_normalization_3/zeros/Const:output:0*
T0*#
_output_shapes
:���������l
)discriminator/layer_normalization_3/ConstConst*
_output_shapes
: *
dtype0*
valueB n
+discriminator/layer_normalization_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
4discriminator/layer_normalization_3/FusedBatchNormV3FusedBatchNormV34discriminator/layer_normalization_3/Reshape:output:01discriminator/layer_normalization_3/ones:output:02discriminator/layer_normalization_3/zeros:output:02discriminator/layer_normalization_3/Const:output:04discriminator/layer_normalization_3/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:��������� :���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
-discriminator/layer_normalization_3/Reshape_1Reshape8discriminator/layer_normalization_3/FusedBatchNormV3:y:02discriminator/layer_normalization_3/Shape:output:0*
T0*/
_output_shapes
:���������< �
8discriminator/layer_normalization_3/mul_4/ReadVariableOpReadVariableOpAdiscriminator_layer_normalization_3_mul_4_readvariableop_resource*
_output_shapes
: *
dtype0�
)discriminator/layer_normalization_3/mul_4Mul6discriminator/layer_normalization_3/Reshape_1:output:0@discriminator/layer_normalization_3/mul_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< �
6discriminator/layer_normalization_3/add/ReadVariableOpReadVariableOp?discriminator_layer_normalization_3_add_readvariableop_resource*
_output_shapes
: *
dtype0�
'discriminator/layer_normalization_3/addAddV2-discriminator/layer_normalization_3/mul_4:z:0>discriminator/layer_normalization_3/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< �
Ediscriminator/spectral_normalization_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOpNdiscriminator_spectral_normalization_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
6discriminator/spectral_normalization_5/conv2d_5/Conv2DConv2D+discriminator/layer_normalization_3/add:z:0Mdiscriminator/spectral_normalization_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<*
paddingSAME*
strides
�
Fdiscriminator/spectral_normalization_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOpOdiscriminator_spectral_normalization_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
7discriminator/spectral_normalization_5/conv2d_5/BiasAddBiasAdd?discriminator/spectral_normalization_5/conv2d_5/Conv2D:output:0Ndiscriminator/spectral_normalization_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<�
Ediscriminator/spectral_normalization_5/conv2d_5/leaky_re_lu/LeakyRelu	LeakyRelu@discriminator/spectral_normalization_5/conv2d_5/BiasAdd:output:0*/
_output_shapes
:���������<�
)discriminator/layer_normalization_4/ShapeShapeSdiscriminator/spectral_normalization_5/conv2d_5/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:�
7discriminator/layer_normalization_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
9discriminator/layer_normalization_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
9discriminator/layer_normalization_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1discriminator/layer_normalization_4/strided_sliceStridedSlice2discriminator/layer_normalization_4/Shape:output:0@discriminator/layer_normalization_4/strided_slice/stack:output:0Bdiscriminator/layer_normalization_4/strided_slice/stack_1:output:0Bdiscriminator/layer_normalization_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)discriminator/layer_normalization_4/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
'discriminator/layer_normalization_4/mulMul2discriminator/layer_normalization_4/mul/x:output:0:discriminator/layer_normalization_4/strided_slice:output:0*
T0*
_output_shapes
: �
9discriminator/layer_normalization_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3discriminator/layer_normalization_4/strided_slice_1StridedSlice2discriminator/layer_normalization_4/Shape:output:0Bdiscriminator/layer_normalization_4/strided_slice_1/stack:output:0Ddiscriminator/layer_normalization_4/strided_slice_1/stack_1:output:0Ddiscriminator/layer_normalization_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
)discriminator/layer_normalization_4/mul_1Mul+discriminator/layer_normalization_4/mul:z:0<discriminator/layer_normalization_4/strided_slice_1:output:0*
T0*
_output_shapes
: �
9discriminator/layer_normalization_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3discriminator/layer_normalization_4/strided_slice_2StridedSlice2discriminator/layer_normalization_4/Shape:output:0Bdiscriminator/layer_normalization_4/strided_slice_2/stack:output:0Ddiscriminator/layer_normalization_4/strided_slice_2/stack_1:output:0Ddiscriminator/layer_normalization_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
)discriminator/layer_normalization_4/mul_2Mul-discriminator/layer_normalization_4/mul_1:z:0<discriminator/layer_normalization_4/strided_slice_2:output:0*
T0*
_output_shapes
: �
9discriminator/layer_normalization_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3discriminator/layer_normalization_4/strided_slice_3StridedSlice2discriminator/layer_normalization_4/Shape:output:0Bdiscriminator/layer_normalization_4/strided_slice_3/stack:output:0Ddiscriminator/layer_normalization_4/strided_slice_3/stack_1:output:0Ddiscriminator/layer_normalization_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+discriminator/layer_normalization_4/mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :�
)discriminator/layer_normalization_4/mul_3Mul4discriminator/layer_normalization_4/mul_3/x:output:0<discriminator/layer_normalization_4/strided_slice_3:output:0*
T0*
_output_shapes
: u
3discriminator/layer_normalization_4/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :u
3discriminator/layer_normalization_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
1discriminator/layer_normalization_4/Reshape/shapePack<discriminator/layer_normalization_4/Reshape/shape/0:output:0-discriminator/layer_normalization_4/mul_2:z:0-discriminator/layer_normalization_4/mul_3:z:0<discriminator/layer_normalization_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
+discriminator/layer_normalization_4/ReshapeReshapeSdiscriminator/spectral_normalization_5/conv2d_5/leaky_re_lu/LeakyRelu:activations:0:discriminator/layer_normalization_4/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
/discriminator/layer_normalization_4/ones/packedPack-discriminator/layer_normalization_4/mul_2:z:0*
N*
T0*
_output_shapes
:s
.discriminator/layer_normalization_4/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
(discriminator/layer_normalization_4/onesFill8discriminator/layer_normalization_4/ones/packed:output:07discriminator/layer_normalization_4/ones/Const:output:0*
T0*#
_output_shapes
:����������
0discriminator/layer_normalization_4/zeros/packedPack-discriminator/layer_normalization_4/mul_2:z:0*
N*
T0*
_output_shapes
:t
/discriminator/layer_normalization_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
)discriminator/layer_normalization_4/zerosFill9discriminator/layer_normalization_4/zeros/packed:output:08discriminator/layer_normalization_4/zeros/Const:output:0*
T0*#
_output_shapes
:���������l
)discriminator/layer_normalization_4/ConstConst*
_output_shapes
: *
dtype0*
valueB n
+discriminator/layer_normalization_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
4discriminator/layer_normalization_4/FusedBatchNormV3FusedBatchNormV34discriminator/layer_normalization_4/Reshape:output:01discriminator/layer_normalization_4/ones:output:02discriminator/layer_normalization_4/zeros:output:02discriminator/layer_normalization_4/Const:output:04discriminator/layer_normalization_4/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
-discriminator/layer_normalization_4/Reshape_1Reshape8discriminator/layer_normalization_4/FusedBatchNormV3:y:02discriminator/layer_normalization_4/Shape:output:0*
T0*/
_output_shapes
:���������<�
8discriminator/layer_normalization_4/mul_4/ReadVariableOpReadVariableOpAdiscriminator_layer_normalization_4_mul_4_readvariableop_resource*
_output_shapes
:*
dtype0�
)discriminator/layer_normalization_4/mul_4Mul6discriminator/layer_normalization_4/Reshape_1:output:0@discriminator/layer_normalization_4/mul_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<�
6discriminator/layer_normalization_4/add/ReadVariableOpReadVariableOp?discriminator_layer_normalization_4_add_readvariableop_resource*
_output_shapes
:*
dtype0�
'discriminator/layer_normalization_4/addAddV2-discriminator/layer_normalization_4/mul_4:z:0>discriminator/layer_normalization_4/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<�
Ediscriminator/spectral_normalization_6/conv2d_6/Conv2D/ReadVariableOpReadVariableOpNdiscriminator_spectral_normalization_6_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
6discriminator/spectral_normalization_6/conv2d_6/Conv2DConv2D+discriminator/layer_normalization_4/add:z:0Mdiscriminator/spectral_normalization_6/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<*
paddingSAME*
strides
�
Fdiscriminator/spectral_normalization_6/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpOdiscriminator_spectral_normalization_6_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
7discriminator/spectral_normalization_6/conv2d_6/BiasAddBiasAdd?discriminator/spectral_normalization_6/conv2d_6/Conv2D:output:0Ndiscriminator/spectral_normalization_6/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<�
Ediscriminator/spectral_normalization_6/conv2d_6/leaky_re_lu/LeakyRelu	LeakyRelu@discriminator/spectral_normalization_6/conv2d_6/BiasAdd:output:0*/
_output_shapes
:���������<�
)discriminator/layer_normalization_5/ShapeShapeSdiscriminator/spectral_normalization_6/conv2d_6/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:�
7discriminator/layer_normalization_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
9discriminator/layer_normalization_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
9discriminator/layer_normalization_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1discriminator/layer_normalization_5/strided_sliceStridedSlice2discriminator/layer_normalization_5/Shape:output:0@discriminator/layer_normalization_5/strided_slice/stack:output:0Bdiscriminator/layer_normalization_5/strided_slice/stack_1:output:0Bdiscriminator/layer_normalization_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)discriminator/layer_normalization_5/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
'discriminator/layer_normalization_5/mulMul2discriminator/layer_normalization_5/mul/x:output:0:discriminator/layer_normalization_5/strided_slice:output:0*
T0*
_output_shapes
: �
9discriminator/layer_normalization_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3discriminator/layer_normalization_5/strided_slice_1StridedSlice2discriminator/layer_normalization_5/Shape:output:0Bdiscriminator/layer_normalization_5/strided_slice_1/stack:output:0Ddiscriminator/layer_normalization_5/strided_slice_1/stack_1:output:0Ddiscriminator/layer_normalization_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
)discriminator/layer_normalization_5/mul_1Mul+discriminator/layer_normalization_5/mul:z:0<discriminator/layer_normalization_5/strided_slice_1:output:0*
T0*
_output_shapes
: �
9discriminator/layer_normalization_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3discriminator/layer_normalization_5/strided_slice_2StridedSlice2discriminator/layer_normalization_5/Shape:output:0Bdiscriminator/layer_normalization_5/strided_slice_2/stack:output:0Ddiscriminator/layer_normalization_5/strided_slice_2/stack_1:output:0Ddiscriminator/layer_normalization_5/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
)discriminator/layer_normalization_5/mul_2Mul-discriminator/layer_normalization_5/mul_1:z:0<discriminator/layer_normalization_5/strided_slice_2:output:0*
T0*
_output_shapes
: �
9discriminator/layer_normalization_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
;discriminator/layer_normalization_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
3discriminator/layer_normalization_5/strided_slice_3StridedSlice2discriminator/layer_normalization_5/Shape:output:0Bdiscriminator/layer_normalization_5/strided_slice_3/stack:output:0Ddiscriminator/layer_normalization_5/strided_slice_3/stack_1:output:0Ddiscriminator/layer_normalization_5/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+discriminator/layer_normalization_5/mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :�
)discriminator/layer_normalization_5/mul_3Mul4discriminator/layer_normalization_5/mul_3/x:output:0<discriminator/layer_normalization_5/strided_slice_3:output:0*
T0*
_output_shapes
: u
3discriminator/layer_normalization_5/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :u
3discriminator/layer_normalization_5/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
1discriminator/layer_normalization_5/Reshape/shapePack<discriminator/layer_normalization_5/Reshape/shape/0:output:0-discriminator/layer_normalization_5/mul_2:z:0-discriminator/layer_normalization_5/mul_3:z:0<discriminator/layer_normalization_5/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
+discriminator/layer_normalization_5/ReshapeReshapeSdiscriminator/spectral_normalization_6/conv2d_6/leaky_re_lu/LeakyRelu:activations:0:discriminator/layer_normalization_5/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
/discriminator/layer_normalization_5/ones/packedPack-discriminator/layer_normalization_5/mul_2:z:0*
N*
T0*
_output_shapes
:s
.discriminator/layer_normalization_5/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
(discriminator/layer_normalization_5/onesFill8discriminator/layer_normalization_5/ones/packed:output:07discriminator/layer_normalization_5/ones/Const:output:0*
T0*#
_output_shapes
:����������
0discriminator/layer_normalization_5/zeros/packedPack-discriminator/layer_normalization_5/mul_2:z:0*
N*
T0*
_output_shapes
:t
/discriminator/layer_normalization_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
)discriminator/layer_normalization_5/zerosFill9discriminator/layer_normalization_5/zeros/packed:output:08discriminator/layer_normalization_5/zeros/Const:output:0*
T0*#
_output_shapes
:���������l
)discriminator/layer_normalization_5/ConstConst*
_output_shapes
: *
dtype0*
valueB n
+discriminator/layer_normalization_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
4discriminator/layer_normalization_5/FusedBatchNormV3FusedBatchNormV34discriminator/layer_normalization_5/Reshape:output:01discriminator/layer_normalization_5/ones:output:02discriminator/layer_normalization_5/zeros:output:02discriminator/layer_normalization_5/Const:output:04discriminator/layer_normalization_5/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
-discriminator/layer_normalization_5/Reshape_1Reshape8discriminator/layer_normalization_5/FusedBatchNormV3:y:02discriminator/layer_normalization_5/Shape:output:0*
T0*/
_output_shapes
:���������<�
8discriminator/layer_normalization_5/mul_4/ReadVariableOpReadVariableOpAdiscriminator_layer_normalization_5_mul_4_readvariableop_resource*
_output_shapes
:*
dtype0�
)discriminator/layer_normalization_5/mul_4Mul6discriminator/layer_normalization_5/Reshape_1:output:0@discriminator/layer_normalization_5/mul_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<�
6discriminator/layer_normalization_5/add/ReadVariableOpReadVariableOp?discriminator_layer_normalization_5_add_readvariableop_resource*
_output_shapes
:*
dtype0�
'discriminator/layer_normalization_5/addAddV2-discriminator/layer_normalization_5/mul_4:z:0>discriminator/layer_normalization_5/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<l
discriminator/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  �
discriminator/flatten/ReshapeReshape+discriminator/layer_normalization_5/add:z:0$discriminator/flatten/Const:output:0*
T0*(
_output_shapes
:�����������
Bdiscriminator/spectral_normalization_7/dense/MatMul/ReadVariableOpReadVariableOpKdiscriminator_spectral_normalization_7_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
3discriminator/spectral_normalization_7/dense/MatMulMatMul&discriminator/flatten/Reshape:output:0Jdiscriminator/spectral_normalization_7/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Cdiscriminator/spectral_normalization_7/dense/BiasAdd/ReadVariableOpReadVariableOpLdiscriminator_spectral_normalization_7_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
4discriminator/spectral_normalization_7/dense/BiasAddBiasAdd=discriminator/spectral_normalization_7/dense/MatMul:product:0Kdiscriminator/spectral_normalization_7/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
IdentityIdentity=discriminator/spectral_normalization_7/dense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp7^discriminator/layer_normalization_3/add/ReadVariableOp9^discriminator/layer_normalization_3/mul_4/ReadVariableOp7^discriminator/layer_normalization_4/add/ReadVariableOp9^discriminator/layer_normalization_4/mul_4/ReadVariableOp7^discriminator/layer_normalization_5/add/ReadVariableOp9^discriminator/layer_normalization_5/mul_4/ReadVariableOpG^discriminator/spectral_normalization_4/conv2d_4/BiasAdd/ReadVariableOpF^discriminator/spectral_normalization_4/conv2d_4/Conv2D/ReadVariableOpG^discriminator/spectral_normalization_5/conv2d_5/BiasAdd/ReadVariableOpF^discriminator/spectral_normalization_5/conv2d_5/Conv2D/ReadVariableOpG^discriminator/spectral_normalization_6/conv2d_6/BiasAdd/ReadVariableOpF^discriminator/spectral_normalization_6/conv2d_6/Conv2D/ReadVariableOpD^discriminator/spectral_normalization_7/dense/BiasAdd/ReadVariableOpC^discriminator/spectral_normalization_7/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������<: : : : : : : : : : : : : : 2p
6discriminator/layer_normalization_3/add/ReadVariableOp6discriminator/layer_normalization_3/add/ReadVariableOp2t
8discriminator/layer_normalization_3/mul_4/ReadVariableOp8discriminator/layer_normalization_3/mul_4/ReadVariableOp2p
6discriminator/layer_normalization_4/add/ReadVariableOp6discriminator/layer_normalization_4/add/ReadVariableOp2t
8discriminator/layer_normalization_4/mul_4/ReadVariableOp8discriminator/layer_normalization_4/mul_4/ReadVariableOp2p
6discriminator/layer_normalization_5/add/ReadVariableOp6discriminator/layer_normalization_5/add/ReadVariableOp2t
8discriminator/layer_normalization_5/mul_4/ReadVariableOp8discriminator/layer_normalization_5/mul_4/ReadVariableOp2�
Fdiscriminator/spectral_normalization_4/conv2d_4/BiasAdd/ReadVariableOpFdiscriminator/spectral_normalization_4/conv2d_4/BiasAdd/ReadVariableOp2�
Ediscriminator/spectral_normalization_4/conv2d_4/Conv2D/ReadVariableOpEdiscriminator/spectral_normalization_4/conv2d_4/Conv2D/ReadVariableOp2�
Fdiscriminator/spectral_normalization_5/conv2d_5/BiasAdd/ReadVariableOpFdiscriminator/spectral_normalization_5/conv2d_5/BiasAdd/ReadVariableOp2�
Ediscriminator/spectral_normalization_5/conv2d_5/Conv2D/ReadVariableOpEdiscriminator/spectral_normalization_5/conv2d_5/Conv2D/ReadVariableOp2�
Fdiscriminator/spectral_normalization_6/conv2d_6/BiasAdd/ReadVariableOpFdiscriminator/spectral_normalization_6/conv2d_6/BiasAdd/ReadVariableOp2�
Ediscriminator/spectral_normalization_6/conv2d_6/Conv2D/ReadVariableOpEdiscriminator/spectral_normalization_6/conv2d_6/Conv2D/ReadVariableOp2�
Cdiscriminator/spectral_normalization_7/dense/BiasAdd/ReadVariableOpCdiscriminator/spectral_normalization_7/dense/BiasAdd/ReadVariableOp2�
Bdiscriminator/spectral_normalization_7/dense/MatMul/ReadVariableOpBdiscriminator/spectral_normalization_7/dense/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������<
!
_user_specified_name	input_2
�4
�
T__inference_spectral_normalization_7_layer_call_and_return_conditional_losses_168909

inputs2
reshape_readvariableop_resource:	�C
1spectral_normalize_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identity��Reshape/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOpw
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:	�*
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   t
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	��
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes
:	�*
transpose_b(
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	�v
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
:	��
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
:	��
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
:	�*
dtype0�
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*
_output_shapes
:	�q
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�     �
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*
_output_shapes
:	��
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
dense/MatMul/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*
_output_shapes
:	�*
dtype0u
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
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
:����������
NoOpNoOp^Reshape/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:����������: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
9__inference_spectral_normalization_5_layer_call_fn_168574

inputs!
unknown:	 
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
GPU2*0J 8� *]
fXRV
T__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_167129w
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
�
�
9__inference_spectral_normalization_7_layer_call_fn_168860

inputs
unknown:	�
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
GPU2*0J 8� *]
fXRV
T__inference_spectral_normalization_7_layer_call_and_return_conditional_losses_167375o
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
:����������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
9__inference_spectral_normalization_6_layer_call_fn_168706

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_167204w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<`
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
T__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_168768

inputs9
reshape_readvariableop_resource:C
1spectral_normalize_matmul_readvariableop_resource:6
(conv2d_6_biasadd_readvariableop_resource:
identity��Reshape/ReadVariableOp�conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:*
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   s
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes

:0�
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes

:0*
transpose_b(~
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes

:0v
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

:0�
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes

:�
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:x
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

:
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes

:0�
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes

:�
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
:*
dtype0�
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:�
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_6/Conv2D/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
:*
dtype0�
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<*
paddingSAME*
strides
�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<w
conv2d_6/leaky_re_lu/LeakyRelu	LeakyReluconv2d_6/BiasAdd:output:0*/
_output_shapes
:���������<�
IdentityIdentity,conv2d_6/leaky_re_lu/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������<�
NoOpNoOp^Reshape/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������<: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�4
�

I__inference_discriminator_layer_call_and_return_conditional_losses_167686

inputs9
spectral_normalization_4_167641: 1
spectral_normalization_4_167643: -
spectral_normalization_4_167645: *
layer_normalization_3_167648: *
layer_normalization_3_167650: 9
spectral_normalization_5_167653:	 1
spectral_normalization_5_167655:-
spectral_normalization_5_167657:*
layer_normalization_4_167660:*
layer_normalization_4_167662:9
spectral_normalization_6_167665:1
spectral_normalization_6_167667:-
spectral_normalization_6_167669:*
layer_normalization_5_167672:*
layer_normalization_5_167674:2
spectral_normalization_7_167678:	�1
spectral_normalization_7_167680:-
spectral_normalization_7_167682:
identity��-layer_normalization_3/StatefulPartitionedCall�-layer_normalization_4/StatefulPartitionedCall�-layer_normalization_5/StatefulPartitionedCall�0spectral_normalization_4/StatefulPartitionedCall�0spectral_normalization_5/StatefulPartitionedCall�0spectral_normalization_6/StatefulPartitionedCall�0spectral_normalization_7/StatefulPartitionedCall�
0spectral_normalization_4/StatefulPartitionedCallStatefulPartitionedCallinputsspectral_normalization_4_167641spectral_normalization_4_167643spectral_normalization_4_167645*
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
GPU2*0J 8� *]
fXRV
T__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_167594�
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCall9spectral_normalization_4/StatefulPartitionedCall:output:0layer_normalization_3_167648layer_normalization_3_167650*
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
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_167112�
0spectral_normalization_5/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0spectral_normalization_5_167653spectral_normalization_5_167655spectral_normalization_5_167657*
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
GPU2*0J 8� *]
fXRV
T__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_167523�
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCall9spectral_normalization_5/StatefulPartitionedCall:output:0layer_normalization_4_167660layer_normalization_4_167662*
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
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_167187�
0spectral_normalization_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0spectral_normalization_6_167665spectral_normalization_6_167667spectral_normalization_6_167669*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_167452�
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCall9spectral_normalization_6/StatefulPartitionedCall:output:0layer_normalization_5_167672layer_normalization_5_167674*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_167262�
flatten/PartitionedCallPartitionedCall6layer_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_167274�
0spectral_normalization_7/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0spectral_normalization_7_167678spectral_normalization_7_167680spectral_normalization_7_167682*
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
GPU2*0J 8� *]
fXRV
T__inference_spectral_normalization_7_layer_call_and_return_conditional_losses_167375�
IdentityIdentity9spectral_normalization_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall.^layer_normalization_5/StatefulPartitionedCall1^spectral_normalization_4/StatefulPartitionedCall1^spectral_normalization_5/StatefulPartitionedCall1^spectral_normalization_6/StatefulPartitionedCall1^spectral_normalization_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������<: : : : : : : : : : : : : : : : : : 2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall2d
0spectral_normalization_4/StatefulPartitionedCall0spectral_normalization_4/StatefulPartitionedCall2d
0spectral_normalization_5/StatefulPartitionedCall0spectral_normalization_5/StatefulPartitionedCall2d
0spectral_normalization_6/StatefulPartitionedCall0spectral_normalization_6/StatefulPartitionedCall2d
0spectral_normalization_7/StatefulPartitionedCall0spectral_normalization_7/StatefulPartitionedCall:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
9__inference_spectral_normalization_4_layer_call_fn_168453

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
identity��StatefulPartitionedCall�
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
GPU2*0J 8� *]
fXRV
T__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_167594w
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
!:���������<: : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
T__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_167054

inputsA
'conv2d_4_conv2d_readvariableop_resource: 6
(conv2d_4_biasadd_readvariableop_resource: 
identity��conv2d_4/BiasAdd/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< *
paddingSAME*
strides
�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< w
conv2d_4/leaky_re_lu/LeakyRelu	LeakyReluconv2d_4/BiasAdd:output:0*/
_output_shapes
:���������< �
IdentityIdentity,conv2d_4/leaky_re_lu/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������< �
NoOpNoOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<: : 2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�1
�
I__inference_discriminator_layer_call_and_return_conditional_losses_167806
input_29
spectral_normalization_4_167769: -
spectral_normalization_4_167771: *
layer_normalization_3_167774: *
layer_normalization_3_167776: 9
spectral_normalization_5_167779:	 -
spectral_normalization_5_167781:*
layer_normalization_4_167784:*
layer_normalization_4_167786:9
spectral_normalization_6_167789:-
spectral_normalization_6_167791:*
layer_normalization_5_167794:*
layer_normalization_5_167796:2
spectral_normalization_7_167800:	�-
spectral_normalization_7_167802:
identity��-layer_normalization_3/StatefulPartitionedCall�-layer_normalization_4/StatefulPartitionedCall�-layer_normalization_5/StatefulPartitionedCall�0spectral_normalization_4/StatefulPartitionedCall�0spectral_normalization_5/StatefulPartitionedCall�0spectral_normalization_6/StatefulPartitionedCall�0spectral_normalization_7/StatefulPartitionedCall�
0spectral_normalization_4/StatefulPartitionedCallStatefulPartitionedCallinput_2spectral_normalization_4_167769spectral_normalization_4_167771*
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
GPU2*0J 8� *]
fXRV
T__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_167054�
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCall9spectral_normalization_4/StatefulPartitionedCall:output:0layer_normalization_3_167774layer_normalization_3_167776*
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
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_167112�
0spectral_normalization_5/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0spectral_normalization_5_167779spectral_normalization_5_167781*
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
GPU2*0J 8� *]
fXRV
T__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_167129�
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCall9spectral_normalization_5/StatefulPartitionedCall:output:0layer_normalization_4_167784layer_normalization_4_167786*
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
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_167187�
0spectral_normalization_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0spectral_normalization_6_167789spectral_normalization_6_167791*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_167204�
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCall9spectral_normalization_6/StatefulPartitionedCall:output:0layer_normalization_5_167794layer_normalization_5_167796*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_167262�
flatten/PartitionedCallPartitionedCall6layer_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_167274�
0spectral_normalization_7/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0spectral_normalization_7_167800spectral_normalization_7_167802*
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
GPU2*0J 8� *]
fXRV
T__inference_spectral_normalization_7_layer_call_and_return_conditional_losses_167286�
IdentityIdentity9spectral_normalization_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall.^layer_normalization_5/StatefulPartitionedCall1^spectral_normalization_4/StatefulPartitionedCall1^spectral_normalization_5/StatefulPartitionedCall1^spectral_normalization_6/StatefulPartitionedCall1^spectral_normalization_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������<: : : : : : : : : : : : : : 2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall2d
0spectral_normalization_4/StatefulPartitionedCall0spectral_normalization_4/StatefulPartitionedCall2d
0spectral_normalization_5/StatefulPartitionedCall0spectral_normalization_5/StatefulPartitionedCall2d
0spectral_normalization_6/StatefulPartitionedCall0spectral_normalization_6/StatefulPartitionedCall2d
0spectral_normalization_7/StatefulPartitionedCall0spectral_normalization_7/StatefulPartitionedCall:X T
/
_output_shapes
:���������<
!
_user_specified_name	input_2
�
�
6__inference_layer_normalization_4_layer_call_fn_168645

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
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_167187w
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
�4
�
T__inference_spectral_normalization_7_layer_call_and_return_conditional_losses_167375

inputs2
reshape_readvariableop_resource:	�C
1spectral_normalize_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identity��Reshape/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOpw
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes
:	�*
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   t
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	��
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes
:	�*
transpose_b(
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	�v
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
:	��
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
:	��
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
:	�*
dtype0�
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*
_output_shapes
:	�q
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�     �
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*
_output_shapes
:	��
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
dense/MatMul/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*
_output_shapes
:	�*
dtype0u
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
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
:����������
NoOpNoOp^Reshape/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:����������: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�)
�
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_168697

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
�/
�	
__inference__traced_save_168986
file_prefix>
:savev2_spectral_normalization_4_kernel_read_readvariableop<
8savev2_spectral_normalization_4_sn_u_read_readvariableop:
6savev2_layer_normalization_3_gamma_read_readvariableop9
5savev2_layer_normalization_3_beta_read_readvariableop>
:savev2_spectral_normalization_5_kernel_read_readvariableop<
8savev2_spectral_normalization_5_sn_u_read_readvariableop:
6savev2_layer_normalization_4_gamma_read_readvariableop9
5savev2_layer_normalization_4_beta_read_readvariableop>
:savev2_spectral_normalization_6_kernel_read_readvariableop<
8savev2_spectral_normalization_6_sn_u_read_readvariableop:
6savev2_layer_normalization_5_gamma_read_readvariableop9
5savev2_layer_normalization_5_beta_read_readvariableop>
:savev2_spectral_normalization_7_kernel_read_readvariableop<
8savev2_spectral_normalization_7_sn_u_read_readvariableop<
8savev2_spectral_normalization_4_bias_read_readvariableop<
8savev2_spectral_normalization_5_bias_read_readvariableop<
8savev2_spectral_normalization_6_bias_read_readvariableop<
8savev2_spectral_normalization_7_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0:savev2_spectral_normalization_4_kernel_read_readvariableop8savev2_spectral_normalization_4_sn_u_read_readvariableop6savev2_layer_normalization_3_gamma_read_readvariableop5savev2_layer_normalization_3_beta_read_readvariableop:savev2_spectral_normalization_5_kernel_read_readvariableop8savev2_spectral_normalization_5_sn_u_read_readvariableop6savev2_layer_normalization_4_gamma_read_readvariableop5savev2_layer_normalization_4_beta_read_readvariableop:savev2_spectral_normalization_6_kernel_read_readvariableop8savev2_spectral_normalization_6_sn_u_read_readvariableop6savev2_layer_normalization_5_gamma_read_readvariableop5savev2_layer_normalization_5_beta_read_readvariableop:savev2_spectral_normalization_7_kernel_read_readvariableop8savev2_spectral_normalization_7_sn_u_read_readvariableop8savev2_spectral_normalization_4_bias_read_readvariableop8savev2_spectral_normalization_5_bias_read_readvariableop8savev2_spectral_normalization_6_bias_read_readvariableop8savev2_spectral_normalization_7_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
�: : : : : :	 ::::::::	�:: :::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:	 :$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
::$
 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	�:$ 

_output_shapes

:: 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
�L
�
"__inference__traced_restore_169050
file_prefixJ
0assignvariableop_spectral_normalization_4_kernel: B
0assignvariableop_1_spectral_normalization_4_sn_u: <
.assignvariableop_2_layer_normalization_3_gamma: ;
-assignvariableop_3_layer_normalization_3_beta: L
2assignvariableop_4_spectral_normalization_5_kernel:	 B
0assignvariableop_5_spectral_normalization_5_sn_u:<
.assignvariableop_6_layer_normalization_4_gamma:;
-assignvariableop_7_layer_normalization_4_beta:L
2assignvariableop_8_spectral_normalization_6_kernel:B
0assignvariableop_9_spectral_normalization_6_sn_u:=
/assignvariableop_10_layer_normalization_5_gamma:<
.assignvariableop_11_layer_normalization_5_beta:F
3assignvariableop_12_spectral_normalization_7_kernel:	�C
1assignvariableop_13_spectral_normalization_7_sn_u:?
1assignvariableop_14_spectral_normalization_4_bias: ?
1assignvariableop_15_spectral_normalization_5_bias:?
1assignvariableop_16_spectral_normalization_6_bias:?
1assignvariableop_17_spectral_normalization_7_bias:
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
AssignVariableOp_2AssignVariableOp.assignvariableop_2_layer_normalization_3_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp-assignvariableop_3_layer_normalization_3_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp2assignvariableop_4_spectral_normalization_5_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp0assignvariableop_5_spectral_normalization_5_sn_uIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp.assignvariableop_6_layer_normalization_4_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp-assignvariableop_7_layer_normalization_4_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp2assignvariableop_8_spectral_normalization_6_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp0assignvariableop_9_spectral_normalization_6_sn_uIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp/assignvariableop_10_layer_normalization_5_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp.assignvariableop_11_layer_normalization_5_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp3assignvariableop_12_spectral_normalization_7_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp1assignvariableop_13_spectral_normalization_7_sn_uIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp1assignvariableop_14_spectral_normalization_4_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp1assignvariableop_15_spectral_normalization_5_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp1assignvariableop_16_spectral_normalization_6_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp1assignvariableop_17_spectral_normalization_7_biasIdentity_17:output:0"/device:CPU:0*
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
�7
�
T__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_168636

inputs9
reshape_readvariableop_resource:	 C
1spectral_normalize_matmul_readvariableop_resource:6
(conv2d_5_biasadd_readvariableop_resource:
identity��Reshape/ReadVariableOp�conv2d_5/BiasAdd/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:	 *
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   t
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	��
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes
:	�*
transpose_b(
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	�v
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
:	��
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

:�
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	��
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
:	 *
dtype0�
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:	 y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   	          �
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:	 �
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_5/Conv2D/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
:	 *
dtype0�
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<*
paddingSAME*
strides
�
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<w
conv2d_5/leaky_re_lu/LeakyRelu	LeakyReluconv2d_5/BiasAdd:output:0*/
_output_shapes
:���������<�
IdentityIdentity,conv2d_5/leaky_re_lu/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������<�
NoOpNoOp^Reshape/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������< : : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2J
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
9__inference_spectral_normalization_4_layer_call_fn_168442

inputs!
unknown: 
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
GPU2*0J 8� *]
fXRV
T__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_167054w
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
:���������<: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_168840

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�)
�
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_167187

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
�)
�
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_168829

inputs+
mul_4_readvariableop_resource:)
add_readvariableop_resource:
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
:���������L
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
[:���������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:t
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*/
_output_shapes
:���������<n
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes
:*
dtype0x
mul_4MulReshape_1:output:0mul_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0m
addAddV2	mul_4:z:0add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<^
IdentityIdentityadd:z:0^NoOp*
T0*/
_output_shapes
:���������<r
NoOpNoOp^add/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�)
�
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_168565

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
�
�
9__inference_spectral_normalization_5_layer_call_fn_168585

inputs!
unknown:	 
	unknown_0:
	unknown_1:
identity��StatefulPartitionedCall�
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
GPU2*0J 8� *]
fXRV
T__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_167523w
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
�
�
9__inference_spectral_normalization_7_layer_call_fn_168849

inputs
unknown:	�
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
GPU2*0J 8� *]
fXRV
T__inference_spectral_normalization_7_layer_call_and_return_conditional_losses_167286o
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
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
T__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_167204

inputsA
'conv2d_6_conv2d_readvariableop_resource:6
(conv2d_6_biasadd_readvariableop_resource:
identity��conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<*
paddingSAME*
strides
�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<w
conv2d_6/leaky_re_lu/LeakyRelu	LeakyReluconv2d_6/BiasAdd:output:0*/
_output_shapes
:���������<�
IdentityIdentity,conv2d_6/leaky_re_lu/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������<�
NoOpNoOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<: : 2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
.__inference_discriminator_layer_call_fn_167766
input_2!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: #
	unknown_4:	 
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:	�

unknown_15:

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
I__inference_discriminator_layer_call_and_return_conditional_losses_167686o
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
_user_specified_name	input_2
�

�
T__inference_spectral_normalization_7_layer_call_and_return_conditional_losses_167286

inputs7
$dense_matmul_readvariableop_resource:	�3
%dense_biasadd_readvariableop_resource:
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0u
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
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
:����������
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
6__inference_layer_normalization_5_layer_call_fn_168777

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_167262w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�7
�
T__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_167523

inputs9
reshape_readvariableop_resource:	 C
1spectral_normalize_matmul_readvariableop_resource:6
(conv2d_5_biasadd_readvariableop_resource:
identity��Reshape/ReadVariableOp�conv2d_5/BiasAdd/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:	 *
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   t
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	��
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes
:	�*
transpose_b(
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	�v
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
:	��
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

:�
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	��
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
:	 *
dtype0�
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:	 y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   	          �
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:	 �
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_5/Conv2D/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
:	 *
dtype0�
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<*
paddingSAME*
strides
�
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<w
conv2d_5/leaky_re_lu/LeakyRelu	LeakyReluconv2d_5/BiasAdd:output:0*/
_output_shapes
:���������<�
IdentityIdentity,conv2d_5/leaky_re_lu/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������<�
NoOpNoOp^Reshape/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������< : : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:���������< 
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_167889
input_2!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: #
	unknown_3:	 
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
!__inference__wrapped_model_167036o
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
_user_specified_name	input_2
�
�
.__inference_discriminator_layer_call_fn_167922

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: #
	unknown_3:	 
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:	�

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
I__inference_discriminator_layer_call_and_return_conditional_losses_167293o
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
�

�
T__inference_spectral_normalization_7_layer_call_and_return_conditional_losses_168870

inputs7
$dense_matmul_readvariableop_resource:	�3
%dense_biasadd_readvariableop_resource:
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0u
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
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
:����������
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�7
�
T__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_167452

inputs9
reshape_readvariableop_resource:C
1spectral_normalize_matmul_readvariableop_resource:6
(conv2d_6_biasadd_readvariableop_resource:
identity��Reshape/ReadVariableOp�conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:*
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   s
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes

:0�
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes

:0*
transpose_b(~
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes

:0v
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

:0�
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes

:�
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:x
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

:
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes

:0�
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes

:�
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
:*
dtype0�
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:�
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_6/Conv2D/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
:*
dtype0�
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<*
paddingSAME*
strides
�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<w
conv2d_6/leaky_re_lu/LeakyRelu	LeakyReluconv2d_6/BiasAdd:output:0*/
_output_shapes
:���������<�
IdentityIdentity,conv2d_6/leaky_re_lu/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������<�
NoOpNoOp^Reshape/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������<: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
.__inference_discriminator_layer_call_fn_167963

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: #
	unknown_4:	 
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:	�

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
I__inference_discriminator_layer_call_and_return_conditional_losses_167686o
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
T__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_168596

inputsA
'conv2d_5_conv2d_readvariableop_resource:	 6
(conv2d_5_biasadd_readvariableop_resource:
identity��conv2d_5/BiasAdd/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<*
paddingSAME*
strides
�
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<w
conv2d_5/leaky_re_lu/LeakyRelu	LeakyReluconv2d_5/BiasAdd:output:0*/
_output_shapes
:���������<�
IdentityIdentity,conv2d_5/leaky_re_lu/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������<�
NoOpNoOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������< : : 2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������< 
 
_user_specified_nameinputs
�
�
.__inference_discriminator_layer_call_fn_167324
input_2!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: #
	unknown_3:	 
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
I__inference_discriminator_layer_call_and_return_conditional_losses_167293o
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
_user_specified_name	input_2
�7
�
T__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_167594

inputs9
reshape_readvariableop_resource: C
1spectral_normalize_matmul_readvariableop_resource: 6
(conv2d_4_biasadd_readvariableop_resource: 
identity��Reshape/ReadVariableOp�conv2d_4/BiasAdd/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
: *
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����    s
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes

:H �
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

: *
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

: 
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes

:H�
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
: *
dtype0�
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
: y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
: �
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_4/Conv2D/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
: *
dtype0�
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< *
paddingSAME*
strides
�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< w
conv2d_4/leaky_re_lu/LeakyRelu	LeakyReluconv2d_4/BiasAdd:output:0*/
_output_shapes
:���������< �
IdentityIdentity,conv2d_4/leaky_re_lu/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������< �
NoOpNoOp^Reshape/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������<: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�7
�
T__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_168504

inputs9
reshape_readvariableop_resource: C
1spectral_normalize_matmul_readvariableop_resource: 6
(conv2d_4_biasadd_readvariableop_resource: 
identity��Reshape/ReadVariableOp�conv2d_4/BiasAdd/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
: *
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����    s
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes

:H �
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

: *
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

: 
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes

:H�
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
: *
dtype0�
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
: y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
: �
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_4/Conv2D/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
: *
dtype0�
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< *
paddingSAME*
strides
�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< w
conv2d_4/leaky_re_lu/LeakyRelu	LeakyReluconv2d_4/BiasAdd:output:0*/
_output_shapes
:���������< �
IdentityIdentity,conv2d_4/leaky_re_lu/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������< �
NoOpNoOp^Reshape/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������<: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�1
�
I__inference_discriminator_layer_call_and_return_conditional_losses_167293

inputs9
spectral_normalization_4_167055: -
spectral_normalization_4_167057: *
layer_normalization_3_167113: *
layer_normalization_3_167115: 9
spectral_normalization_5_167130:	 -
spectral_normalization_5_167132:*
layer_normalization_4_167188:*
layer_normalization_4_167190:9
spectral_normalization_6_167205:-
spectral_normalization_6_167207:*
layer_normalization_5_167263:*
layer_normalization_5_167265:2
spectral_normalization_7_167287:	�-
spectral_normalization_7_167289:
identity��-layer_normalization_3/StatefulPartitionedCall�-layer_normalization_4/StatefulPartitionedCall�-layer_normalization_5/StatefulPartitionedCall�0spectral_normalization_4/StatefulPartitionedCall�0spectral_normalization_5/StatefulPartitionedCall�0spectral_normalization_6/StatefulPartitionedCall�0spectral_normalization_7/StatefulPartitionedCall�
0spectral_normalization_4/StatefulPartitionedCallStatefulPartitionedCallinputsspectral_normalization_4_167055spectral_normalization_4_167057*
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
GPU2*0J 8� *]
fXRV
T__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_167054�
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCall9spectral_normalization_4/StatefulPartitionedCall:output:0layer_normalization_3_167113layer_normalization_3_167115*
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
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_167112�
0spectral_normalization_5/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0spectral_normalization_5_167130spectral_normalization_5_167132*
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
GPU2*0J 8� *]
fXRV
T__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_167129�
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCall9spectral_normalization_5/StatefulPartitionedCall:output:0layer_normalization_4_167188layer_normalization_4_167190*
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
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_167187�
0spectral_normalization_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0spectral_normalization_6_167205spectral_normalization_6_167207*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_167204�
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCall9spectral_normalization_6/StatefulPartitionedCall:output:0layer_normalization_5_167263layer_normalization_5_167265*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_167262�
flatten/PartitionedCallPartitionedCall6layer_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_167274�
0spectral_normalization_7/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0spectral_normalization_7_167287spectral_normalization_7_167289*
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
GPU2*0J 8� *]
fXRV
T__inference_spectral_normalization_7_layer_call_and_return_conditional_losses_167286�
IdentityIdentity9spectral_normalization_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall.^layer_normalization_5/StatefulPartitionedCall1^spectral_normalization_4/StatefulPartitionedCall1^spectral_normalization_5/StatefulPartitionedCall1^spectral_normalization_6/StatefulPartitionedCall1^spectral_normalization_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������<: : : : : : : : : : : : : : 2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall2d
0spectral_normalization_4/StatefulPartitionedCall0spectral_normalization_4/StatefulPartitionedCall2d
0spectral_normalization_5/StatefulPartitionedCall0spectral_normalization_5/StatefulPartitionedCall2d
0spectral_normalization_6/StatefulPartitionedCall0spectral_normalization_6/StatefulPartitionedCall2d
0spectral_normalization_7/StatefulPartitionedCall0spectral_normalization_7/StatefulPartitionedCall:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
T__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_167129

inputsA
'conv2d_5_conv2d_readvariableop_resource:	 6
(conv2d_5_biasadd_readvariableop_resource:
identity��conv2d_5/BiasAdd/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<*
paddingSAME*
strides
�
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<w
conv2d_5/leaky_re_lu/LeakyRelu	LeakyReluconv2d_5/BiasAdd:output:0*/
_output_shapes
:���������<�
IdentityIdentity,conv2d_5/leaky_re_lu/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������<�
NoOpNoOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������< : : 2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������< 
 
_user_specified_nameinputs
�
D
(__inference_flatten_layer_call_fn_168834

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
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_167274a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
��
�
I__inference_discriminator_layer_call_and_return_conditional_losses_168433

inputsR
8spectral_normalization_4_reshape_readvariableop_resource: \
Jspectral_normalization_4_spectral_normalize_matmul_readvariableop_resource: O
Aspectral_normalization_4_conv2d_4_biasadd_readvariableop_resource: A
3layer_normalization_3_mul_4_readvariableop_resource: ?
1layer_normalization_3_add_readvariableop_resource: R
8spectral_normalization_5_reshape_readvariableop_resource:	 \
Jspectral_normalization_5_spectral_normalize_matmul_readvariableop_resource:O
Aspectral_normalization_5_conv2d_5_biasadd_readvariableop_resource:A
3layer_normalization_4_mul_4_readvariableop_resource:?
1layer_normalization_4_add_readvariableop_resource:R
8spectral_normalization_6_reshape_readvariableop_resource:\
Jspectral_normalization_6_spectral_normalize_matmul_readvariableop_resource:O
Aspectral_normalization_6_conv2d_6_biasadd_readvariableop_resource:A
3layer_normalization_5_mul_4_readvariableop_resource:?
1layer_normalization_5_add_readvariableop_resource:K
8spectral_normalization_7_reshape_readvariableop_resource:	�\
Jspectral_normalization_7_spectral_normalize_matmul_readvariableop_resource:L
>spectral_normalization_7_dense_biasadd_readvariableop_resource:
identity��(layer_normalization_3/add/ReadVariableOp�*layer_normalization_3/mul_4/ReadVariableOp�(layer_normalization_4/add/ReadVariableOp�*layer_normalization_4/mul_4/ReadVariableOp�(layer_normalization_5/add/ReadVariableOp�*layer_normalization_5/mul_4/ReadVariableOp�/spectral_normalization_4/Reshape/ReadVariableOp�8spectral_normalization_4/conv2d_4/BiasAdd/ReadVariableOp�7spectral_normalization_4/conv2d_4/Conv2D/ReadVariableOp�<spectral_normalization_4/spectral_normalize/AssignVariableOp�>spectral_normalization_4/spectral_normalize/AssignVariableOp_1�Aspectral_normalization_4/spectral_normalize/MatMul/ReadVariableOp�:spectral_normalization_4/spectral_normalize/ReadVariableOp�/spectral_normalization_5/Reshape/ReadVariableOp�8spectral_normalization_5/conv2d_5/BiasAdd/ReadVariableOp�7spectral_normalization_5/conv2d_5/Conv2D/ReadVariableOp�<spectral_normalization_5/spectral_normalize/AssignVariableOp�>spectral_normalization_5/spectral_normalize/AssignVariableOp_1�Aspectral_normalization_5/spectral_normalize/MatMul/ReadVariableOp�:spectral_normalization_5/spectral_normalize/ReadVariableOp�/spectral_normalization_6/Reshape/ReadVariableOp�8spectral_normalization_6/conv2d_6/BiasAdd/ReadVariableOp�7spectral_normalization_6/conv2d_6/Conv2D/ReadVariableOp�<spectral_normalization_6/spectral_normalize/AssignVariableOp�>spectral_normalization_6/spectral_normalize/AssignVariableOp_1�Aspectral_normalization_6/spectral_normalize/MatMul/ReadVariableOp�:spectral_normalization_6/spectral_normalize/ReadVariableOp�/spectral_normalization_7/Reshape/ReadVariableOp�5spectral_normalization_7/dense/BiasAdd/ReadVariableOp�4spectral_normalization_7/dense/MatMul/ReadVariableOp�<spectral_normalization_7/spectral_normalize/AssignVariableOp�>spectral_normalization_7/spectral_normalize/AssignVariableOp_1�Aspectral_normalization_7/spectral_normalize/MatMul/ReadVariableOp�:spectral_normalization_7/spectral_normalize/ReadVariableOp�
/spectral_normalization_4/Reshape/ReadVariableOpReadVariableOp8spectral_normalization_4_reshape_readvariableop_resource*&
_output_shapes
: *
dtype0w
&spectral_normalization_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����    �
 spectral_normalization_4/ReshapeReshape7spectral_normalization_4/Reshape/ReadVariableOp:value:0/spectral_normalization_4/Reshape/shape:output:0*
T0*
_output_shapes

:H �
Aspectral_normalization_4/spectral_normalize/MatMul/ReadVariableOpReadVariableOpJspectral_normalization_4_spectral_normalize_matmul_readvariableop_resource*
_output_shapes

: *
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

: �
Aspectral_normalization_4/spectral_normalize/l2_normalize_1/SquareSquare>spectral_normalization_4/spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

: �
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

: �
8spectral_normalization_4/spectral_normalize/StopGradientStopGradient>spectral_normalization_4/spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

: �
:spectral_normalization_4/spectral_normalize/StopGradient_1StopGradient<spectral_normalization_4/spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes

:H�
4spectral_normalization_4/spectral_normalize/MatMul_2MatMulCspectral_normalization_4/spectral_normalize/StopGradient_1:output:0)spectral_normalization_4/Reshape:output:0*
T0*
_output_shapes

: �
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
: *
dtype0�
3spectral_normalization_4/spectral_normalize/truedivRealDivBspectral_normalization_4/spectral_normalize/ReadVariableOp:value:0>spectral_normalization_4/spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
: �
9spectral_normalization_4/spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
3spectral_normalization_4/spectral_normalize/ReshapeReshape7spectral_normalization_4/spectral_normalize/truediv:z:0Bspectral_normalization_4/spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
: �
>spectral_normalization_4/spectral_normalize/AssignVariableOp_1AssignVariableOp8spectral_normalization_4_reshape_readvariableop_resource<spectral_normalization_4/spectral_normalize/Reshape:output:00^spectral_normalization_4/Reshape/ReadVariableOp;^spectral_normalization_4/spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
7spectral_normalization_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp8spectral_normalization_4_reshape_readvariableop_resource?^spectral_normalization_4/spectral_normalize/AssignVariableOp_1*&
_output_shapes
: *
dtype0�
(spectral_normalization_4/conv2d_4/Conv2DConv2Dinputs?spectral_normalization_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< *
paddingSAME*
strides
�
8spectral_normalization_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOpAspectral_normalization_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
)spectral_normalization_4/conv2d_4/BiasAddBiasAdd1spectral_normalization_4/conv2d_4/Conv2D:output:0@spectral_normalization_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< �
7spectral_normalization_4/conv2d_4/leaky_re_lu/LeakyRelu	LeakyRelu2spectral_normalization_4/conv2d_4/BiasAdd:output:0*/
_output_shapes
:���������< �
layer_normalization_3/ShapeShapeEspectral_normalization_4/conv2d_4/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:s
)layer_normalization_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_3/strided_sliceStridedSlice$layer_normalization_3/Shape:output:02layer_normalization_3/strided_slice/stack:output:04layer_normalization_3/strided_slice/stack_1:output:04layer_normalization_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_3/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_3/mulMul$layer_normalization_3/mul/x:output:0,layer_normalization_3/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_3/strided_slice_1StridedSlice$layer_normalization_3/Shape:output:04layer_normalization_3/strided_slice_1/stack:output:06layer_normalization_3/strided_slice_1/stack_1:output:06layer_normalization_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_3/mul_1Mullayer_normalization_3/mul:z:0.layer_normalization_3/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_3/strided_slice_2StridedSlice$layer_normalization_3/Shape:output:04layer_normalization_3/strided_slice_2/stack:output:06layer_normalization_3/strided_slice_2/stack_1:output:06layer_normalization_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_3/mul_2Mullayer_normalization_3/mul_1:z:0.layer_normalization_3/strided_slice_2:output:0*
T0*
_output_shapes
: u
+layer_normalization_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_3/strided_slice_3StridedSlice$layer_normalization_3/Shape:output:04layer_normalization_3/strided_slice_3/stack:output:06layer_normalization_3/strided_slice_3/stack_1:output:06layer_normalization_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_3/mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_3/mul_3Mul&layer_normalization_3/mul_3/x:output:0.layer_normalization_3/strided_slice_3:output:0*
T0*
_output_shapes
: g
%layer_normalization_3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
#layer_normalization_3/Reshape/shapePack.layer_normalization_3/Reshape/shape/0:output:0layer_normalization_3/mul_2:z:0layer_normalization_3/mul_3:z:0.layer_normalization_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
layer_normalization_3/ReshapeReshapeEspectral_normalization_4/conv2d_4/leaky_re_lu/LeakyRelu:activations:0,layer_normalization_3/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� x
!layer_normalization_3/ones/packedPacklayer_normalization_3/mul_2:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
layer_normalization_3/onesFill*layer_normalization_3/ones/packed:output:0)layer_normalization_3/ones/Const:output:0*
T0*#
_output_shapes
:���������y
"layer_normalization_3/zeros/packedPacklayer_normalization_3/mul_2:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
layer_normalization_3/zerosFill+layer_normalization_3/zeros/packed:output:0*layer_normalization_3/zeros/Const:output:0*
T0*#
_output_shapes
:���������^
layer_normalization_3/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
&layer_normalization_3/FusedBatchNormV3FusedBatchNormV3&layer_normalization_3/Reshape:output:0#layer_normalization_3/ones:output:0$layer_normalization_3/zeros:output:0$layer_normalization_3/Const:output:0&layer_normalization_3/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:��������� :���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
layer_normalization_3/Reshape_1Reshape*layer_normalization_3/FusedBatchNormV3:y:0$layer_normalization_3/Shape:output:0*
T0*/
_output_shapes
:���������< �
*layer_normalization_3/mul_4/ReadVariableOpReadVariableOp3layer_normalization_3_mul_4_readvariableop_resource*
_output_shapes
: *
dtype0�
layer_normalization_3/mul_4Mul(layer_normalization_3/Reshape_1:output:02layer_normalization_3/mul_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< �
(layer_normalization_3/add/ReadVariableOpReadVariableOp1layer_normalization_3_add_readvariableop_resource*
_output_shapes
: *
dtype0�
layer_normalization_3/addAddV2layer_normalization_3/mul_4:z:00layer_normalization_3/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< �
/spectral_normalization_5/Reshape/ReadVariableOpReadVariableOp8spectral_normalization_5_reshape_readvariableop_resource*&
_output_shapes
:	 *
dtype0w
&spectral_normalization_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
 spectral_normalization_5/ReshapeReshape7spectral_normalization_5/Reshape/ReadVariableOp:value:0/spectral_normalization_5/Reshape/shape:output:0*
T0*
_output_shapes
:	��
Aspectral_normalization_5/spectral_normalize/MatMul/ReadVariableOpReadVariableOpJspectral_normalization_5_spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
2spectral_normalization_5/spectral_normalize/MatMulMatMulIspectral_normalization_5/spectral_normalize/MatMul/ReadVariableOp:value:0)spectral_normalization_5/Reshape:output:0*
T0*
_output_shapes
:	�*
transpose_b(�
?spectral_normalization_5/spectral_normalize/l2_normalize/SquareSquare<spectral_normalization_5/spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	��
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
:	��
4spectral_normalization_5/spectral_normalize/MatMul_1MatMul<spectral_normalization_5/spectral_normalize/l2_normalize:z:0)spectral_normalization_5/Reshape:output:0*
T0*
_output_shapes

:�
Aspectral_normalization_5/spectral_normalize/l2_normalize_1/SquareSquare>spectral_normalization_5/spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:�
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

:�
8spectral_normalization_5/spectral_normalize/StopGradientStopGradient>spectral_normalization_5/spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:�
:spectral_normalization_5/spectral_normalize/StopGradient_1StopGradient<spectral_normalization_5/spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	��
4spectral_normalization_5/spectral_normalize/MatMul_2MatMulCspectral_normalization_5/spectral_normalize/StopGradient_1:output:0)spectral_normalization_5/Reshape:output:0*
T0*
_output_shapes

:�
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
:	 *
dtype0�
3spectral_normalization_5/spectral_normalize/truedivRealDivBspectral_normalization_5/spectral_normalize/ReadVariableOp:value:0>spectral_normalization_5/spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:	 �
9spectral_normalization_5/spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   	          �
3spectral_normalization_5/spectral_normalize/ReshapeReshape7spectral_normalization_5/spectral_normalize/truediv:z:0Bspectral_normalization_5/spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:	 �
>spectral_normalization_5/spectral_normalize/AssignVariableOp_1AssignVariableOp8spectral_normalization_5_reshape_readvariableop_resource<spectral_normalization_5/spectral_normalize/Reshape:output:00^spectral_normalization_5/Reshape/ReadVariableOp;^spectral_normalization_5/spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
7spectral_normalization_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp8spectral_normalization_5_reshape_readvariableop_resource?^spectral_normalization_5/spectral_normalize/AssignVariableOp_1*&
_output_shapes
:	 *
dtype0�
(spectral_normalization_5/conv2d_5/Conv2DConv2Dlayer_normalization_3/add:z:0?spectral_normalization_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<*
paddingSAME*
strides
�
8spectral_normalization_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOpAspectral_normalization_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
)spectral_normalization_5/conv2d_5/BiasAddBiasAdd1spectral_normalization_5/conv2d_5/Conv2D:output:0@spectral_normalization_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<�
7spectral_normalization_5/conv2d_5/leaky_re_lu/LeakyRelu	LeakyRelu2spectral_normalization_5/conv2d_5/BiasAdd:output:0*/
_output_shapes
:���������<�
layer_normalization_4/ShapeShapeEspectral_normalization_5/conv2d_5/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:s
)layer_normalization_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_4/strided_sliceStridedSlice$layer_normalization_4/Shape:output:02layer_normalization_4/strided_slice/stack:output:04layer_normalization_4/strided_slice/stack_1:output:04layer_normalization_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_4/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_4/mulMul$layer_normalization_4/mul/x:output:0,layer_normalization_4/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_4/strided_slice_1StridedSlice$layer_normalization_4/Shape:output:04layer_normalization_4/strided_slice_1/stack:output:06layer_normalization_4/strided_slice_1/stack_1:output:06layer_normalization_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_4/mul_1Mullayer_normalization_4/mul:z:0.layer_normalization_4/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_4/strided_slice_2StridedSlice$layer_normalization_4/Shape:output:04layer_normalization_4/strided_slice_2/stack:output:06layer_normalization_4/strided_slice_2/stack_1:output:06layer_normalization_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_4/mul_2Mullayer_normalization_4/mul_1:z:0.layer_normalization_4/strided_slice_2:output:0*
T0*
_output_shapes
: u
+layer_normalization_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_4/strided_slice_3StridedSlice$layer_normalization_4/Shape:output:04layer_normalization_4/strided_slice_3/stack:output:06layer_normalization_4/strided_slice_3/stack_1:output:06layer_normalization_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_4/mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_4/mul_3Mul&layer_normalization_4/mul_3/x:output:0.layer_normalization_4/strided_slice_3:output:0*
T0*
_output_shapes
: g
%layer_normalization_4/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
#layer_normalization_4/Reshape/shapePack.layer_normalization_4/Reshape/shape/0:output:0layer_normalization_4/mul_2:z:0layer_normalization_4/mul_3:z:0.layer_normalization_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
layer_normalization_4/ReshapeReshapeEspectral_normalization_5/conv2d_5/leaky_re_lu/LeakyRelu:activations:0,layer_normalization_4/Reshape/shape:output:0*
T0*/
_output_shapes
:���������x
!layer_normalization_4/ones/packedPacklayer_normalization_4/mul_2:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_4/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
layer_normalization_4/onesFill*layer_normalization_4/ones/packed:output:0)layer_normalization_4/ones/Const:output:0*
T0*#
_output_shapes
:���������y
"layer_normalization_4/zeros/packedPacklayer_normalization_4/mul_2:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
layer_normalization_4/zerosFill+layer_normalization_4/zeros/packed:output:0*layer_normalization_4/zeros/Const:output:0*
T0*#
_output_shapes
:���������^
layer_normalization_4/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
&layer_normalization_4/FusedBatchNormV3FusedBatchNormV3&layer_normalization_4/Reshape:output:0#layer_normalization_4/ones:output:0$layer_normalization_4/zeros:output:0$layer_normalization_4/Const:output:0&layer_normalization_4/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
layer_normalization_4/Reshape_1Reshape*layer_normalization_4/FusedBatchNormV3:y:0$layer_normalization_4/Shape:output:0*
T0*/
_output_shapes
:���������<�
*layer_normalization_4/mul_4/ReadVariableOpReadVariableOp3layer_normalization_4_mul_4_readvariableop_resource*
_output_shapes
:*
dtype0�
layer_normalization_4/mul_4Mul(layer_normalization_4/Reshape_1:output:02layer_normalization_4/mul_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<�
(layer_normalization_4/add/ReadVariableOpReadVariableOp1layer_normalization_4_add_readvariableop_resource*
_output_shapes
:*
dtype0�
layer_normalization_4/addAddV2layer_normalization_4/mul_4:z:00layer_normalization_4/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<�
/spectral_normalization_6/Reshape/ReadVariableOpReadVariableOp8spectral_normalization_6_reshape_readvariableop_resource*&
_output_shapes
:*
dtype0w
&spectral_normalization_6/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
 spectral_normalization_6/ReshapeReshape7spectral_normalization_6/Reshape/ReadVariableOp:value:0/spectral_normalization_6/Reshape/shape:output:0*
T0*
_output_shapes

:0�
Aspectral_normalization_6/spectral_normalize/MatMul/ReadVariableOpReadVariableOpJspectral_normalization_6_spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
2spectral_normalization_6/spectral_normalize/MatMulMatMulIspectral_normalization_6/spectral_normalize/MatMul/ReadVariableOp:value:0)spectral_normalization_6/Reshape:output:0*
T0*
_output_shapes

:0*
transpose_b(�
?spectral_normalization_6/spectral_normalize/l2_normalize/SquareSquare<spectral_normalization_6/spectral_normalize/MatMul:product:0*
T0*
_output_shapes

:0�
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

:0�
4spectral_normalization_6/spectral_normalize/MatMul_1MatMul<spectral_normalization_6/spectral_normalize/l2_normalize:z:0)spectral_normalization_6/Reshape:output:0*
T0*
_output_shapes

:�
Aspectral_normalization_6/spectral_normalize/l2_normalize_1/SquareSquare>spectral_normalization_6/spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:�
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

:�
8spectral_normalization_6/spectral_normalize/StopGradientStopGradient>spectral_normalization_6/spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:�
:spectral_normalization_6/spectral_normalize/StopGradient_1StopGradient<spectral_normalization_6/spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes

:0�
4spectral_normalization_6/spectral_normalize/MatMul_2MatMulCspectral_normalization_6/spectral_normalize/StopGradient_1:output:0)spectral_normalization_6/Reshape:output:0*
T0*
_output_shapes

:�
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
:*
dtype0�
3spectral_normalization_6/spectral_normalize/truedivRealDivBspectral_normalization_6/spectral_normalize/ReadVariableOp:value:0>spectral_normalization_6/spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:�
9spectral_normalization_6/spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
3spectral_normalization_6/spectral_normalize/ReshapeReshape7spectral_normalization_6/spectral_normalize/truediv:z:0Bspectral_normalization_6/spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:�
>spectral_normalization_6/spectral_normalize/AssignVariableOp_1AssignVariableOp8spectral_normalization_6_reshape_readvariableop_resource<spectral_normalization_6/spectral_normalize/Reshape:output:00^spectral_normalization_6/Reshape/ReadVariableOp;^spectral_normalization_6/spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
7spectral_normalization_6/conv2d_6/Conv2D/ReadVariableOpReadVariableOp8spectral_normalization_6_reshape_readvariableop_resource?^spectral_normalization_6/spectral_normalize/AssignVariableOp_1*&
_output_shapes
:*
dtype0�
(spectral_normalization_6/conv2d_6/Conv2DConv2Dlayer_normalization_4/add:z:0?spectral_normalization_6/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<*
paddingSAME*
strides
�
8spectral_normalization_6/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpAspectral_normalization_6_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
)spectral_normalization_6/conv2d_6/BiasAddBiasAdd1spectral_normalization_6/conv2d_6/Conv2D:output:0@spectral_normalization_6/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<�
7spectral_normalization_6/conv2d_6/leaky_re_lu/LeakyRelu	LeakyRelu2spectral_normalization_6/conv2d_6/BiasAdd:output:0*/
_output_shapes
:���������<�
layer_normalization_5/ShapeShapeEspectral_normalization_6/conv2d_6/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:s
)layer_normalization_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_5/strided_sliceStridedSlice$layer_normalization_5/Shape:output:02layer_normalization_5/strided_slice/stack:output:04layer_normalization_5/strided_slice/stack_1:output:04layer_normalization_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_5/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_5/mulMul$layer_normalization_5/mul/x:output:0,layer_normalization_5/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_5/strided_slice_1StridedSlice$layer_normalization_5/Shape:output:04layer_normalization_5/strided_slice_1/stack:output:06layer_normalization_5/strided_slice_1/stack_1:output:06layer_normalization_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_5/mul_1Mullayer_normalization_5/mul:z:0.layer_normalization_5/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_5/strided_slice_2StridedSlice$layer_normalization_5/Shape:output:04layer_normalization_5/strided_slice_2/stack:output:06layer_normalization_5/strided_slice_2/stack_1:output:06layer_normalization_5/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_5/mul_2Mullayer_normalization_5/mul_1:z:0.layer_normalization_5/strided_slice_2:output:0*
T0*
_output_shapes
: u
+layer_normalization_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_5/strided_slice_3StridedSlice$layer_normalization_5/Shape:output:04layer_normalization_5/strided_slice_3/stack:output:06layer_normalization_5/strided_slice_3/stack_1:output:06layer_normalization_5/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_5/mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_5/mul_3Mul&layer_normalization_5/mul_3/x:output:0.layer_normalization_5/strided_slice_3:output:0*
T0*
_output_shapes
: g
%layer_normalization_5/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_5/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
#layer_normalization_5/Reshape/shapePack.layer_normalization_5/Reshape/shape/0:output:0layer_normalization_5/mul_2:z:0layer_normalization_5/mul_3:z:0.layer_normalization_5/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
layer_normalization_5/ReshapeReshapeEspectral_normalization_6/conv2d_6/leaky_re_lu/LeakyRelu:activations:0,layer_normalization_5/Reshape/shape:output:0*
T0*/
_output_shapes
:���������x
!layer_normalization_5/ones/packedPacklayer_normalization_5/mul_2:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_5/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
layer_normalization_5/onesFill*layer_normalization_5/ones/packed:output:0)layer_normalization_5/ones/Const:output:0*
T0*#
_output_shapes
:���������y
"layer_normalization_5/zeros/packedPacklayer_normalization_5/mul_2:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
layer_normalization_5/zerosFill+layer_normalization_5/zeros/packed:output:0*layer_normalization_5/zeros/Const:output:0*
T0*#
_output_shapes
:���������^
layer_normalization_5/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
&layer_normalization_5/FusedBatchNormV3FusedBatchNormV3&layer_normalization_5/Reshape:output:0#layer_normalization_5/ones:output:0$layer_normalization_5/zeros:output:0$layer_normalization_5/Const:output:0&layer_normalization_5/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
layer_normalization_5/Reshape_1Reshape*layer_normalization_5/FusedBatchNormV3:y:0$layer_normalization_5/Shape:output:0*
T0*/
_output_shapes
:���������<�
*layer_normalization_5/mul_4/ReadVariableOpReadVariableOp3layer_normalization_5_mul_4_readvariableop_resource*
_output_shapes
:*
dtype0�
layer_normalization_5/mul_4Mul(layer_normalization_5/Reshape_1:output:02layer_normalization_5/mul_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<�
(layer_normalization_5/add/ReadVariableOpReadVariableOp1layer_normalization_5_add_readvariableop_resource*
_output_shapes
:*
dtype0�
layer_normalization_5/addAddV2layer_normalization_5/mul_4:z:00layer_normalization_5/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  �
flatten/ReshapeReshapelayer_normalization_5/add:z:0flatten/Const:output:0*
T0*(
_output_shapes
:�����������
/spectral_normalization_7/Reshape/ReadVariableOpReadVariableOp8spectral_normalization_7_reshape_readvariableop_resource*
_output_shapes
:	�*
dtype0w
&spectral_normalization_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
 spectral_normalization_7/ReshapeReshape7spectral_normalization_7/Reshape/ReadVariableOp:value:0/spectral_normalization_7/Reshape/shape:output:0*
T0*
_output_shapes
:	��
Aspectral_normalization_7/spectral_normalize/MatMul/ReadVariableOpReadVariableOpJspectral_normalization_7_spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
2spectral_normalization_7/spectral_normalize/MatMulMatMulIspectral_normalization_7/spectral_normalize/MatMul/ReadVariableOp:value:0)spectral_normalization_7/Reshape:output:0*
T0*
_output_shapes
:	�*
transpose_b(�
?spectral_normalization_7/spectral_normalize/l2_normalize/SquareSquare<spectral_normalization_7/spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	��
>spectral_normalization_7/spectral_normalize/l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
<spectral_normalization_7/spectral_normalize/l2_normalize/SumSumCspectral_normalization_7/spectral_normalize/l2_normalize/Square:y:0Gspectral_normalization_7/spectral_normalize/l2_normalize/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(�
Bspectral_normalization_7/spectral_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
@spectral_normalization_7/spectral_normalize/l2_normalize/MaximumMaximumEspectral_normalization_7/spectral_normalize/l2_normalize/Sum:output:0Kspectral_normalization_7/spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:�
>spectral_normalization_7/spectral_normalize/l2_normalize/RsqrtRsqrtDspectral_normalization_7/spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:�
8spectral_normalization_7/spectral_normalize/l2_normalizeMul<spectral_normalization_7/spectral_normalize/MatMul:product:0Bspectral_normalization_7/spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes
:	��
4spectral_normalization_7/spectral_normalize/MatMul_1MatMul<spectral_normalization_7/spectral_normalize/l2_normalize:z:0)spectral_normalization_7/Reshape:output:0*
T0*
_output_shapes

:�
Aspectral_normalization_7/spectral_normalize/l2_normalize_1/SquareSquare>spectral_normalization_7/spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:�
@spectral_normalization_7/spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
>spectral_normalization_7/spectral_normalize/l2_normalize_1/SumSumEspectral_normalization_7/spectral_normalize/l2_normalize_1/Square:y:0Ispectral_normalization_7/spectral_normalize/l2_normalize_1/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(�
Dspectral_normalization_7/spectral_normalize/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
Bspectral_normalization_7/spectral_normalize/l2_normalize_1/MaximumMaximumGspectral_normalization_7/spectral_normalize/l2_normalize_1/Sum:output:0Mspectral_normalization_7/spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:�
@spectral_normalization_7/spectral_normalize/l2_normalize_1/RsqrtRsqrtFspectral_normalization_7/spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:�
:spectral_normalization_7/spectral_normalize/l2_normalize_1Mul>spectral_normalization_7/spectral_normalize/MatMul_1:product:0Dspectral_normalization_7/spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

:�
8spectral_normalization_7/spectral_normalize/StopGradientStopGradient>spectral_normalization_7/spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:�
:spectral_normalization_7/spectral_normalize/StopGradient_1StopGradient<spectral_normalization_7/spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	��
4spectral_normalization_7/spectral_normalize/MatMul_2MatMulCspectral_normalization_7/spectral_normalize/StopGradient_1:output:0)spectral_normalization_7/Reshape:output:0*
T0*
_output_shapes

:�
4spectral_normalization_7/spectral_normalize/MatMul_3MatMul>spectral_normalization_7/spectral_normalize/MatMul_2:product:0Aspectral_normalization_7/spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(�
<spectral_normalization_7/spectral_normalize/AssignVariableOpAssignVariableOpJspectral_normalization_7_spectral_normalize_matmul_readvariableop_resourceAspectral_normalization_7/spectral_normalize/StopGradient:output:0B^spectral_normalization_7/spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
:spectral_normalization_7/spectral_normalize/ReadVariableOpReadVariableOp8spectral_normalization_7_reshape_readvariableop_resource*
_output_shapes
:	�*
dtype0�
3spectral_normalization_7/spectral_normalize/truedivRealDivBspectral_normalization_7/spectral_normalize/ReadVariableOp:value:0>spectral_normalization_7/spectral_normalize/MatMul_3:product:0*
T0*
_output_shapes
:	��
9spectral_normalization_7/spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�     �
3spectral_normalization_7/spectral_normalize/ReshapeReshape7spectral_normalization_7/spectral_normalize/truediv:z:0Bspectral_normalization_7/spectral_normalize/Reshape/shape:output:0*
T0*
_output_shapes
:	��
>spectral_normalization_7/spectral_normalize/AssignVariableOp_1AssignVariableOp8spectral_normalization_7_reshape_readvariableop_resource<spectral_normalization_7/spectral_normalize/Reshape:output:00^spectral_normalization_7/Reshape/ReadVariableOp;^spectral_normalization_7/spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
4spectral_normalization_7/dense/MatMul/ReadVariableOpReadVariableOp8spectral_normalization_7_reshape_readvariableop_resource?^spectral_normalization_7/spectral_normalize/AssignVariableOp_1*
_output_shapes
:	�*
dtype0�
%spectral_normalization_7/dense/MatMulMatMulflatten/Reshape:output:0<spectral_normalization_7/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
5spectral_normalization_7/dense/BiasAdd/ReadVariableOpReadVariableOp>spectral_normalization_7_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
&spectral_normalization_7/dense/BiasAddBiasAdd/spectral_normalization_7/dense/MatMul:product:0=spectral_normalization_7/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
IdentityIdentity/spectral_normalization_7/dense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp)^layer_normalization_3/add/ReadVariableOp+^layer_normalization_3/mul_4/ReadVariableOp)^layer_normalization_4/add/ReadVariableOp+^layer_normalization_4/mul_4/ReadVariableOp)^layer_normalization_5/add/ReadVariableOp+^layer_normalization_5/mul_4/ReadVariableOp0^spectral_normalization_4/Reshape/ReadVariableOp9^spectral_normalization_4/conv2d_4/BiasAdd/ReadVariableOp8^spectral_normalization_4/conv2d_4/Conv2D/ReadVariableOp=^spectral_normalization_4/spectral_normalize/AssignVariableOp?^spectral_normalization_4/spectral_normalize/AssignVariableOp_1B^spectral_normalization_4/spectral_normalize/MatMul/ReadVariableOp;^spectral_normalization_4/spectral_normalize/ReadVariableOp0^spectral_normalization_5/Reshape/ReadVariableOp9^spectral_normalization_5/conv2d_5/BiasAdd/ReadVariableOp8^spectral_normalization_5/conv2d_5/Conv2D/ReadVariableOp=^spectral_normalization_5/spectral_normalize/AssignVariableOp?^spectral_normalization_5/spectral_normalize/AssignVariableOp_1B^spectral_normalization_5/spectral_normalize/MatMul/ReadVariableOp;^spectral_normalization_5/spectral_normalize/ReadVariableOp0^spectral_normalization_6/Reshape/ReadVariableOp9^spectral_normalization_6/conv2d_6/BiasAdd/ReadVariableOp8^spectral_normalization_6/conv2d_6/Conv2D/ReadVariableOp=^spectral_normalization_6/spectral_normalize/AssignVariableOp?^spectral_normalization_6/spectral_normalize/AssignVariableOp_1B^spectral_normalization_6/spectral_normalize/MatMul/ReadVariableOp;^spectral_normalization_6/spectral_normalize/ReadVariableOp0^spectral_normalization_7/Reshape/ReadVariableOp6^spectral_normalization_7/dense/BiasAdd/ReadVariableOp5^spectral_normalization_7/dense/MatMul/ReadVariableOp=^spectral_normalization_7/spectral_normalize/AssignVariableOp?^spectral_normalization_7/spectral_normalize/AssignVariableOp_1B^spectral_normalization_7/spectral_normalize/MatMul/ReadVariableOp;^spectral_normalization_7/spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������<: : : : : : : : : : : : : : : : : : 2T
(layer_normalization_3/add/ReadVariableOp(layer_normalization_3/add/ReadVariableOp2X
*layer_normalization_3/mul_4/ReadVariableOp*layer_normalization_3/mul_4/ReadVariableOp2T
(layer_normalization_4/add/ReadVariableOp(layer_normalization_4/add/ReadVariableOp2X
*layer_normalization_4/mul_4/ReadVariableOp*layer_normalization_4/mul_4/ReadVariableOp2T
(layer_normalization_5/add/ReadVariableOp(layer_normalization_5/add/ReadVariableOp2X
*layer_normalization_5/mul_4/ReadVariableOp*layer_normalization_5/mul_4/ReadVariableOp2b
/spectral_normalization_4/Reshape/ReadVariableOp/spectral_normalization_4/Reshape/ReadVariableOp2t
8spectral_normalization_4/conv2d_4/BiasAdd/ReadVariableOp8spectral_normalization_4/conv2d_4/BiasAdd/ReadVariableOp2r
7spectral_normalization_4/conv2d_4/Conv2D/ReadVariableOp7spectral_normalization_4/conv2d_4/Conv2D/ReadVariableOp2|
<spectral_normalization_4/spectral_normalize/AssignVariableOp<spectral_normalization_4/spectral_normalize/AssignVariableOp2�
>spectral_normalization_4/spectral_normalize/AssignVariableOp_1>spectral_normalization_4/spectral_normalize/AssignVariableOp_12�
Aspectral_normalization_4/spectral_normalize/MatMul/ReadVariableOpAspectral_normalization_4/spectral_normalize/MatMul/ReadVariableOp2x
:spectral_normalization_4/spectral_normalize/ReadVariableOp:spectral_normalization_4/spectral_normalize/ReadVariableOp2b
/spectral_normalization_5/Reshape/ReadVariableOp/spectral_normalization_5/Reshape/ReadVariableOp2t
8spectral_normalization_5/conv2d_5/BiasAdd/ReadVariableOp8spectral_normalization_5/conv2d_5/BiasAdd/ReadVariableOp2r
7spectral_normalization_5/conv2d_5/Conv2D/ReadVariableOp7spectral_normalization_5/conv2d_5/Conv2D/ReadVariableOp2|
<spectral_normalization_5/spectral_normalize/AssignVariableOp<spectral_normalization_5/spectral_normalize/AssignVariableOp2�
>spectral_normalization_5/spectral_normalize/AssignVariableOp_1>spectral_normalization_5/spectral_normalize/AssignVariableOp_12�
Aspectral_normalization_5/spectral_normalize/MatMul/ReadVariableOpAspectral_normalization_5/spectral_normalize/MatMul/ReadVariableOp2x
:spectral_normalization_5/spectral_normalize/ReadVariableOp:spectral_normalization_5/spectral_normalize/ReadVariableOp2b
/spectral_normalization_6/Reshape/ReadVariableOp/spectral_normalization_6/Reshape/ReadVariableOp2t
8spectral_normalization_6/conv2d_6/BiasAdd/ReadVariableOp8spectral_normalization_6/conv2d_6/BiasAdd/ReadVariableOp2r
7spectral_normalization_6/conv2d_6/Conv2D/ReadVariableOp7spectral_normalization_6/conv2d_6/Conv2D/ReadVariableOp2|
<spectral_normalization_6/spectral_normalize/AssignVariableOp<spectral_normalization_6/spectral_normalize/AssignVariableOp2�
>spectral_normalization_6/spectral_normalize/AssignVariableOp_1>spectral_normalization_6/spectral_normalize/AssignVariableOp_12�
Aspectral_normalization_6/spectral_normalize/MatMul/ReadVariableOpAspectral_normalization_6/spectral_normalize/MatMul/ReadVariableOp2x
:spectral_normalization_6/spectral_normalize/ReadVariableOp:spectral_normalization_6/spectral_normalize/ReadVariableOp2b
/spectral_normalization_7/Reshape/ReadVariableOp/spectral_normalization_7/Reshape/ReadVariableOp2n
5spectral_normalization_7/dense/BiasAdd/ReadVariableOp5spectral_normalization_7/dense/BiasAdd/ReadVariableOp2l
4spectral_normalization_7/dense/MatMul/ReadVariableOp4spectral_normalization_7/dense/MatMul/ReadVariableOp2|
<spectral_normalization_7/spectral_normalize/AssignVariableOp<spectral_normalization_7/spectral_normalize/AssignVariableOp2�
>spectral_normalization_7/spectral_normalize/AssignVariableOp_1>spectral_normalization_7/spectral_normalize/AssignVariableOp_12�
Aspectral_normalization_7/spectral_normalize/MatMul/ReadVariableOpAspectral_normalization_7/spectral_normalize/MatMul/ReadVariableOp2x
:spectral_normalization_7/spectral_normalize/ReadVariableOp:spectral_normalization_7/spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�4
�

I__inference_discriminator_layer_call_and_return_conditional_losses_167854
input_29
spectral_normalization_4_167809: 1
spectral_normalization_4_167811: -
spectral_normalization_4_167813: *
layer_normalization_3_167816: *
layer_normalization_3_167818: 9
spectral_normalization_5_167821:	 1
spectral_normalization_5_167823:-
spectral_normalization_5_167825:*
layer_normalization_4_167828:*
layer_normalization_4_167830:9
spectral_normalization_6_167833:1
spectral_normalization_6_167835:-
spectral_normalization_6_167837:*
layer_normalization_5_167840:*
layer_normalization_5_167842:2
spectral_normalization_7_167846:	�1
spectral_normalization_7_167848:-
spectral_normalization_7_167850:
identity��-layer_normalization_3/StatefulPartitionedCall�-layer_normalization_4/StatefulPartitionedCall�-layer_normalization_5/StatefulPartitionedCall�0spectral_normalization_4/StatefulPartitionedCall�0spectral_normalization_5/StatefulPartitionedCall�0spectral_normalization_6/StatefulPartitionedCall�0spectral_normalization_7/StatefulPartitionedCall�
0spectral_normalization_4/StatefulPartitionedCallStatefulPartitionedCallinput_2spectral_normalization_4_167809spectral_normalization_4_167811spectral_normalization_4_167813*
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
GPU2*0J 8� *]
fXRV
T__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_167594�
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCall9spectral_normalization_4/StatefulPartitionedCall:output:0layer_normalization_3_167816layer_normalization_3_167818*
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
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_167112�
0spectral_normalization_5/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0spectral_normalization_5_167821spectral_normalization_5_167823spectral_normalization_5_167825*
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
GPU2*0J 8� *]
fXRV
T__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_167523�
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCall9spectral_normalization_5/StatefulPartitionedCall:output:0layer_normalization_4_167828layer_normalization_4_167830*
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
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_167187�
0spectral_normalization_6/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0spectral_normalization_6_167833spectral_normalization_6_167835spectral_normalization_6_167837*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_167452�
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCall9spectral_normalization_6/StatefulPartitionedCall:output:0layer_normalization_5_167840layer_normalization_5_167842*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_167262�
flatten/PartitionedCallPartitionedCall6layer_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_167274�
0spectral_normalization_7/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0spectral_normalization_7_167846spectral_normalization_7_167848spectral_normalization_7_167850*
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
GPU2*0J 8� *]
fXRV
T__inference_spectral_normalization_7_layer_call_and_return_conditional_losses_167375�
IdentityIdentity9spectral_normalization_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall.^layer_normalization_5/StatefulPartitionedCall1^spectral_normalization_4/StatefulPartitionedCall1^spectral_normalization_5/StatefulPartitionedCall1^spectral_normalization_6/StatefulPartitionedCall1^spectral_normalization_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������<: : : : : : : : : : : : : : : : : : 2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall2d
0spectral_normalization_4/StatefulPartitionedCall0spectral_normalization_4/StatefulPartitionedCall2d
0spectral_normalization_5/StatefulPartitionedCall0spectral_normalization_5/StatefulPartitionedCall2d
0spectral_normalization_6/StatefulPartitionedCall0spectral_normalization_6/StatefulPartitionedCall2d
0spectral_normalization_7/StatefulPartitionedCall0spectral_normalization_7/StatefulPartitionedCall:X T
/
_output_shapes
:���������<
!
_user_specified_name	input_2
�
�
T__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_168728

inputsA
'conv2d_6_conv2d_readvariableop_resource:6
(conv2d_6_biasadd_readvariableop_resource:
identity��conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<*
paddingSAME*
strides
�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<w
conv2d_6/leaky_re_lu/LeakyRelu	LeakyReluconv2d_6/BiasAdd:output:0*/
_output_shapes
:���������<�
IdentityIdentity,conv2d_6/leaky_re_lu/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������<�
NoOpNoOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<: : 2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
9__inference_spectral_normalization_6_layer_call_fn_168717

inputs!
unknown:
	unknown_0:
	unknown_1:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_167452w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������<: : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
��
�
I__inference_discriminator_layer_call_and_return_conditional_losses_168140

inputsZ
@spectral_normalization_4_conv2d_4_conv2d_readvariableop_resource: O
Aspectral_normalization_4_conv2d_4_biasadd_readvariableop_resource: A
3layer_normalization_3_mul_4_readvariableop_resource: ?
1layer_normalization_3_add_readvariableop_resource: Z
@spectral_normalization_5_conv2d_5_conv2d_readvariableop_resource:	 O
Aspectral_normalization_5_conv2d_5_biasadd_readvariableop_resource:A
3layer_normalization_4_mul_4_readvariableop_resource:?
1layer_normalization_4_add_readvariableop_resource:Z
@spectral_normalization_6_conv2d_6_conv2d_readvariableop_resource:O
Aspectral_normalization_6_conv2d_6_biasadd_readvariableop_resource:A
3layer_normalization_5_mul_4_readvariableop_resource:?
1layer_normalization_5_add_readvariableop_resource:P
=spectral_normalization_7_dense_matmul_readvariableop_resource:	�L
>spectral_normalization_7_dense_biasadd_readvariableop_resource:
identity��(layer_normalization_3/add/ReadVariableOp�*layer_normalization_3/mul_4/ReadVariableOp�(layer_normalization_4/add/ReadVariableOp�*layer_normalization_4/mul_4/ReadVariableOp�(layer_normalization_5/add/ReadVariableOp�*layer_normalization_5/mul_4/ReadVariableOp�8spectral_normalization_4/conv2d_4/BiasAdd/ReadVariableOp�7spectral_normalization_4/conv2d_4/Conv2D/ReadVariableOp�8spectral_normalization_5/conv2d_5/BiasAdd/ReadVariableOp�7spectral_normalization_5/conv2d_5/Conv2D/ReadVariableOp�8spectral_normalization_6/conv2d_6/BiasAdd/ReadVariableOp�7spectral_normalization_6/conv2d_6/Conv2D/ReadVariableOp�5spectral_normalization_7/dense/BiasAdd/ReadVariableOp�4spectral_normalization_7/dense/MatMul/ReadVariableOp�
7spectral_normalization_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp@spectral_normalization_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
(spectral_normalization_4/conv2d_4/Conv2DConv2Dinputs?spectral_normalization_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< *
paddingSAME*
strides
�
8spectral_normalization_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOpAspectral_normalization_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
)spectral_normalization_4/conv2d_4/BiasAddBiasAdd1spectral_normalization_4/conv2d_4/Conv2D:output:0@spectral_normalization_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< �
7spectral_normalization_4/conv2d_4/leaky_re_lu/LeakyRelu	LeakyRelu2spectral_normalization_4/conv2d_4/BiasAdd:output:0*/
_output_shapes
:���������< �
layer_normalization_3/ShapeShapeEspectral_normalization_4/conv2d_4/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:s
)layer_normalization_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_3/strided_sliceStridedSlice$layer_normalization_3/Shape:output:02layer_normalization_3/strided_slice/stack:output:04layer_normalization_3/strided_slice/stack_1:output:04layer_normalization_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_3/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_3/mulMul$layer_normalization_3/mul/x:output:0,layer_normalization_3/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_3/strided_slice_1StridedSlice$layer_normalization_3/Shape:output:04layer_normalization_3/strided_slice_1/stack:output:06layer_normalization_3/strided_slice_1/stack_1:output:06layer_normalization_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_3/mul_1Mullayer_normalization_3/mul:z:0.layer_normalization_3/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_3/strided_slice_2StridedSlice$layer_normalization_3/Shape:output:04layer_normalization_3/strided_slice_2/stack:output:06layer_normalization_3/strided_slice_2/stack_1:output:06layer_normalization_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_3/mul_2Mullayer_normalization_3/mul_1:z:0.layer_normalization_3/strided_slice_2:output:0*
T0*
_output_shapes
: u
+layer_normalization_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_3/strided_slice_3StridedSlice$layer_normalization_3/Shape:output:04layer_normalization_3/strided_slice_3/stack:output:06layer_normalization_3/strided_slice_3/stack_1:output:06layer_normalization_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_3/mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_3/mul_3Mul&layer_normalization_3/mul_3/x:output:0.layer_normalization_3/strided_slice_3:output:0*
T0*
_output_shapes
: g
%layer_normalization_3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
#layer_normalization_3/Reshape/shapePack.layer_normalization_3/Reshape/shape/0:output:0layer_normalization_3/mul_2:z:0layer_normalization_3/mul_3:z:0.layer_normalization_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
layer_normalization_3/ReshapeReshapeEspectral_normalization_4/conv2d_4/leaky_re_lu/LeakyRelu:activations:0,layer_normalization_3/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� x
!layer_normalization_3/ones/packedPacklayer_normalization_3/mul_2:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
layer_normalization_3/onesFill*layer_normalization_3/ones/packed:output:0)layer_normalization_3/ones/Const:output:0*
T0*#
_output_shapes
:���������y
"layer_normalization_3/zeros/packedPacklayer_normalization_3/mul_2:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
layer_normalization_3/zerosFill+layer_normalization_3/zeros/packed:output:0*layer_normalization_3/zeros/Const:output:0*
T0*#
_output_shapes
:���������^
layer_normalization_3/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
&layer_normalization_3/FusedBatchNormV3FusedBatchNormV3&layer_normalization_3/Reshape:output:0#layer_normalization_3/ones:output:0$layer_normalization_3/zeros:output:0$layer_normalization_3/Const:output:0&layer_normalization_3/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:��������� :���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
layer_normalization_3/Reshape_1Reshape*layer_normalization_3/FusedBatchNormV3:y:0$layer_normalization_3/Shape:output:0*
T0*/
_output_shapes
:���������< �
*layer_normalization_3/mul_4/ReadVariableOpReadVariableOp3layer_normalization_3_mul_4_readvariableop_resource*
_output_shapes
: *
dtype0�
layer_normalization_3/mul_4Mul(layer_normalization_3/Reshape_1:output:02layer_normalization_3/mul_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< �
(layer_normalization_3/add/ReadVariableOpReadVariableOp1layer_normalization_3_add_readvariableop_resource*
_output_shapes
: *
dtype0�
layer_normalization_3/addAddV2layer_normalization_3/mul_4:z:00layer_normalization_3/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< �
7spectral_normalization_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp@spectral_normalization_5_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:	 *
dtype0�
(spectral_normalization_5/conv2d_5/Conv2DConv2Dlayer_normalization_3/add:z:0?spectral_normalization_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<*
paddingSAME*
strides
�
8spectral_normalization_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOpAspectral_normalization_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
)spectral_normalization_5/conv2d_5/BiasAddBiasAdd1spectral_normalization_5/conv2d_5/Conv2D:output:0@spectral_normalization_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<�
7spectral_normalization_5/conv2d_5/leaky_re_lu/LeakyRelu	LeakyRelu2spectral_normalization_5/conv2d_5/BiasAdd:output:0*/
_output_shapes
:���������<�
layer_normalization_4/ShapeShapeEspectral_normalization_5/conv2d_5/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:s
)layer_normalization_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_4/strided_sliceStridedSlice$layer_normalization_4/Shape:output:02layer_normalization_4/strided_slice/stack:output:04layer_normalization_4/strided_slice/stack_1:output:04layer_normalization_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_4/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_4/mulMul$layer_normalization_4/mul/x:output:0,layer_normalization_4/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_4/strided_slice_1StridedSlice$layer_normalization_4/Shape:output:04layer_normalization_4/strided_slice_1/stack:output:06layer_normalization_4/strided_slice_1/stack_1:output:06layer_normalization_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_4/mul_1Mullayer_normalization_4/mul:z:0.layer_normalization_4/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_4/strided_slice_2StridedSlice$layer_normalization_4/Shape:output:04layer_normalization_4/strided_slice_2/stack:output:06layer_normalization_4/strided_slice_2/stack_1:output:06layer_normalization_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_4/mul_2Mullayer_normalization_4/mul_1:z:0.layer_normalization_4/strided_slice_2:output:0*
T0*
_output_shapes
: u
+layer_normalization_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_4/strided_slice_3StridedSlice$layer_normalization_4/Shape:output:04layer_normalization_4/strided_slice_3/stack:output:06layer_normalization_4/strided_slice_3/stack_1:output:06layer_normalization_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_4/mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_4/mul_3Mul&layer_normalization_4/mul_3/x:output:0.layer_normalization_4/strided_slice_3:output:0*
T0*
_output_shapes
: g
%layer_normalization_4/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
#layer_normalization_4/Reshape/shapePack.layer_normalization_4/Reshape/shape/0:output:0layer_normalization_4/mul_2:z:0layer_normalization_4/mul_3:z:0.layer_normalization_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
layer_normalization_4/ReshapeReshapeEspectral_normalization_5/conv2d_5/leaky_re_lu/LeakyRelu:activations:0,layer_normalization_4/Reshape/shape:output:0*
T0*/
_output_shapes
:���������x
!layer_normalization_4/ones/packedPacklayer_normalization_4/mul_2:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_4/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
layer_normalization_4/onesFill*layer_normalization_4/ones/packed:output:0)layer_normalization_4/ones/Const:output:0*
T0*#
_output_shapes
:���������y
"layer_normalization_4/zeros/packedPacklayer_normalization_4/mul_2:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
layer_normalization_4/zerosFill+layer_normalization_4/zeros/packed:output:0*layer_normalization_4/zeros/Const:output:0*
T0*#
_output_shapes
:���������^
layer_normalization_4/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
&layer_normalization_4/FusedBatchNormV3FusedBatchNormV3&layer_normalization_4/Reshape:output:0#layer_normalization_4/ones:output:0$layer_normalization_4/zeros:output:0$layer_normalization_4/Const:output:0&layer_normalization_4/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
layer_normalization_4/Reshape_1Reshape*layer_normalization_4/FusedBatchNormV3:y:0$layer_normalization_4/Shape:output:0*
T0*/
_output_shapes
:���������<�
*layer_normalization_4/mul_4/ReadVariableOpReadVariableOp3layer_normalization_4_mul_4_readvariableop_resource*
_output_shapes
:*
dtype0�
layer_normalization_4/mul_4Mul(layer_normalization_4/Reshape_1:output:02layer_normalization_4/mul_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<�
(layer_normalization_4/add/ReadVariableOpReadVariableOp1layer_normalization_4_add_readvariableop_resource*
_output_shapes
:*
dtype0�
layer_normalization_4/addAddV2layer_normalization_4/mul_4:z:00layer_normalization_4/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<�
7spectral_normalization_6/conv2d_6/Conv2D/ReadVariableOpReadVariableOp@spectral_normalization_6_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
(spectral_normalization_6/conv2d_6/Conv2DConv2Dlayer_normalization_4/add:z:0?spectral_normalization_6/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<*
paddingSAME*
strides
�
8spectral_normalization_6/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpAspectral_normalization_6_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
)spectral_normalization_6/conv2d_6/BiasAddBiasAdd1spectral_normalization_6/conv2d_6/Conv2D:output:0@spectral_normalization_6/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<�
7spectral_normalization_6/conv2d_6/leaky_re_lu/LeakyRelu	LeakyRelu2spectral_normalization_6/conv2d_6/BiasAdd:output:0*/
_output_shapes
:���������<�
layer_normalization_5/ShapeShapeEspectral_normalization_6/conv2d_6/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:s
)layer_normalization_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#layer_normalization_5/strided_sliceStridedSlice$layer_normalization_5/Shape:output:02layer_normalization_5/strided_slice/stack:output:04layer_normalization_5/strided_slice/stack_1:output:04layer_normalization_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_5/mul/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_5/mulMul$layer_normalization_5/mul/x:output:0,layer_normalization_5/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_5/strided_slice_1StridedSlice$layer_normalization_5/Shape:output:04layer_normalization_5/strided_slice_1/stack:output:06layer_normalization_5/strided_slice_1/stack_1:output:06layer_normalization_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_5/mul_1Mullayer_normalization_5/mul:z:0.layer_normalization_5/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_5/strided_slice_2StridedSlice$layer_normalization_5/Shape:output:04layer_normalization_5/strided_slice_2/stack:output:06layer_normalization_5/strided_slice_2/stack_1:output:06layer_normalization_5/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
layer_normalization_5/mul_2Mullayer_normalization_5/mul_1:z:0.layer_normalization_5/strided_slice_2:output:0*
T0*
_output_shapes
: u
+layer_normalization_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%layer_normalization_5/strided_slice_3StridedSlice$layer_normalization_5/Shape:output:04layer_normalization_5/strided_slice_3/stack:output:06layer_normalization_5/strided_slice_3/stack_1:output:06layer_normalization_5/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_5/mul_3/xConst*
_output_shapes
: *
dtype0*
value	B :�
layer_normalization_5/mul_3Mul&layer_normalization_5/mul_3/x:output:0.layer_normalization_5/strided_slice_3:output:0*
T0*
_output_shapes
: g
%layer_normalization_5/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_5/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
#layer_normalization_5/Reshape/shapePack.layer_normalization_5/Reshape/shape/0:output:0layer_normalization_5/mul_2:z:0layer_normalization_5/mul_3:z:0.layer_normalization_5/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
layer_normalization_5/ReshapeReshapeEspectral_normalization_6/conv2d_6/leaky_re_lu/LeakyRelu:activations:0,layer_normalization_5/Reshape/shape:output:0*
T0*/
_output_shapes
:���������x
!layer_normalization_5/ones/packedPacklayer_normalization_5/mul_2:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_5/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
layer_normalization_5/onesFill*layer_normalization_5/ones/packed:output:0)layer_normalization_5/ones/Const:output:0*
T0*#
_output_shapes
:���������y
"layer_normalization_5/zeros/packedPacklayer_normalization_5/mul_2:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
layer_normalization_5/zerosFill+layer_normalization_5/zeros/packed:output:0*layer_normalization_5/zeros/Const:output:0*
T0*#
_output_shapes
:���������^
layer_normalization_5/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB �
&layer_normalization_5/FusedBatchNormV3FusedBatchNormV3&layer_normalization_5/Reshape:output:0#layer_normalization_5/ones:output:0$layer_normalization_5/zeros:output:0$layer_normalization_5/Const:output:0&layer_normalization_5/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:���������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:�
layer_normalization_5/Reshape_1Reshape*layer_normalization_5/FusedBatchNormV3:y:0$layer_normalization_5/Shape:output:0*
T0*/
_output_shapes
:���������<�
*layer_normalization_5/mul_4/ReadVariableOpReadVariableOp3layer_normalization_5_mul_4_readvariableop_resource*
_output_shapes
:*
dtype0�
layer_normalization_5/mul_4Mul(layer_normalization_5/Reshape_1:output:02layer_normalization_5/mul_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<�
(layer_normalization_5/add/ReadVariableOpReadVariableOp1layer_normalization_5_add_readvariableop_resource*
_output_shapes
:*
dtype0�
layer_normalization_5/addAddV2layer_normalization_5/mul_4:z:00layer_normalization_5/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  �
flatten/ReshapeReshapelayer_normalization_5/add:z:0flatten/Const:output:0*
T0*(
_output_shapes
:�����������
4spectral_normalization_7/dense/MatMul/ReadVariableOpReadVariableOp=spectral_normalization_7_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
%spectral_normalization_7/dense/MatMulMatMulflatten/Reshape:output:0<spectral_normalization_7/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
5spectral_normalization_7/dense/BiasAdd/ReadVariableOpReadVariableOp>spectral_normalization_7_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
&spectral_normalization_7/dense/BiasAddBiasAdd/spectral_normalization_7/dense/MatMul:product:0=spectral_normalization_7/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
IdentityIdentity/spectral_normalization_7/dense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp)^layer_normalization_3/add/ReadVariableOp+^layer_normalization_3/mul_4/ReadVariableOp)^layer_normalization_4/add/ReadVariableOp+^layer_normalization_4/mul_4/ReadVariableOp)^layer_normalization_5/add/ReadVariableOp+^layer_normalization_5/mul_4/ReadVariableOp9^spectral_normalization_4/conv2d_4/BiasAdd/ReadVariableOp8^spectral_normalization_4/conv2d_4/Conv2D/ReadVariableOp9^spectral_normalization_5/conv2d_5/BiasAdd/ReadVariableOp8^spectral_normalization_5/conv2d_5/Conv2D/ReadVariableOp9^spectral_normalization_6/conv2d_6/BiasAdd/ReadVariableOp8^spectral_normalization_6/conv2d_6/Conv2D/ReadVariableOp6^spectral_normalization_7/dense/BiasAdd/ReadVariableOp5^spectral_normalization_7/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������<: : : : : : : : : : : : : : 2T
(layer_normalization_3/add/ReadVariableOp(layer_normalization_3/add/ReadVariableOp2X
*layer_normalization_3/mul_4/ReadVariableOp*layer_normalization_3/mul_4/ReadVariableOp2T
(layer_normalization_4/add/ReadVariableOp(layer_normalization_4/add/ReadVariableOp2X
*layer_normalization_4/mul_4/ReadVariableOp*layer_normalization_4/mul_4/ReadVariableOp2T
(layer_normalization_5/add/ReadVariableOp(layer_normalization_5/add/ReadVariableOp2X
*layer_normalization_5/mul_4/ReadVariableOp*layer_normalization_5/mul_4/ReadVariableOp2t
8spectral_normalization_4/conv2d_4/BiasAdd/ReadVariableOp8spectral_normalization_4/conv2d_4/BiasAdd/ReadVariableOp2r
7spectral_normalization_4/conv2d_4/Conv2D/ReadVariableOp7spectral_normalization_4/conv2d_4/Conv2D/ReadVariableOp2t
8spectral_normalization_5/conv2d_5/BiasAdd/ReadVariableOp8spectral_normalization_5/conv2d_5/BiasAdd/ReadVariableOp2r
7spectral_normalization_5/conv2d_5/Conv2D/ReadVariableOp7spectral_normalization_5/conv2d_5/Conv2D/ReadVariableOp2t
8spectral_normalization_6/conv2d_6/BiasAdd/ReadVariableOp8spectral_normalization_6/conv2d_6/BiasAdd/ReadVariableOp2r
7spectral_normalization_6/conv2d_6/Conv2D/ReadVariableOp7spectral_normalization_6/conv2d_6/Conv2D/ReadVariableOp2n
5spectral_normalization_7/dense/BiasAdd/ReadVariableOp5spectral_normalization_7/dense/BiasAdd/ReadVariableOp2l
4spectral_normalization_7/dense/MatMul/ReadVariableOp4spectral_normalization_7/dense/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�)
�
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_167112

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
T__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_168464

inputsA
'conv2d_4_conv2d_readvariableop_resource: 6
(conv2d_4_biasadd_readvariableop_resource: 
identity��conv2d_4/BiasAdd/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< *
paddingSAME*
strides
�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������< w
conv2d_4/leaky_re_lu/LeakyRelu	LeakyReluconv2d_4/BiasAdd:output:0*/
_output_shapes
:���������< �
IdentityIdentity,conv2d_4/leaky_re_lu/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������< �
NoOpNoOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<: : 2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�)
�
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_167262

inputs+
mul_4_readvariableop_resource:)
add_readvariableop_resource:
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
:���������L
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
[:���������:���������:���������:���������:���������:*
data_formatNCHW*
epsilon%o�:t
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*/
_output_shapes
:���������<n
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes
:*
dtype0x
mul_4MulReshape_1:output:0mul_4/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0m
addAddV2	mul_4:z:0add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<^
IdentityIdentityadd:z:0^NoOp*
T0*/
_output_shapes
:���������<r
NoOpNoOp^add/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������<: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:W S
/
_output_shapes
:���������<
 
_user_specified_nameinputs
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_167274

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<:W S
/
_output_shapes
:���������<
 
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
input_28
serving_default_input_2:0���������<L
spectral_normalization_70
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
.__inference_discriminator_layer_call_fn_167324
.__inference_discriminator_layer_call_fn_167922
.__inference_discriminator_layer_call_fn_167963
.__inference_discriminator_layer_call_fn_167766�
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
I__inference_discriminator_layer_call_and_return_conditional_losses_168140
I__inference_discriminator_layer_call_and_return_conditional_losses_168433
I__inference_discriminator_layer_call_and_return_conditional_losses_167806
I__inference_discriminator_layer_call_and_return_conditional_losses_167854�
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
!__inference__wrapped_model_167036input_2"�
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
9__inference_spectral_normalization_4_layer_call_fn_168442
9__inference_spectral_normalization_4_layer_call_fn_168453�
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
T__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_168464
T__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_168504�
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
9:7 2spectral_normalization_4/kernel
 "
trackable_list_wrapper
-:+ 2spectral_normalization_4/sn_u
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
6__inference_layer_normalization_3_layer_call_fn_168513�
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
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_168565�
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
):' 2layer_normalization_3/gamma
(:& 2layer_normalization_3/beta
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
9__inference_spectral_normalization_5_layer_call_fn_168574
9__inference_spectral_normalization_5_layer_call_fn_168585�
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
T__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_168596
T__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_168636�
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
9:7	 2spectral_normalization_5/kernel
 "
trackable_list_wrapper
-:+2spectral_normalization_5/sn_u
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
6__inference_layer_normalization_4_layer_call_fn_168645�
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
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_168697�
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
):'2layer_normalization_4/gamma
(:&2layer_normalization_4/beta
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
9__inference_spectral_normalization_6_layer_call_fn_168706
9__inference_spectral_normalization_6_layer_call_fn_168717�
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
T__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_168728
T__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_168768�
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
9:72spectral_normalization_6/kernel
 "
trackable_list_wrapper
-:+2spectral_normalization_6/sn_u
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
6__inference_layer_normalization_5_layer_call_fn_168777�
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
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_168829�
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
):'2layer_normalization_5/gamma
(:&2layer_normalization_5/beta
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
(__inference_flatten_layer_call_fn_168834�
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
C__inference_flatten_layer_call_and_return_conditional_losses_168840�
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
9__inference_spectral_normalization_7_layer_call_fn_168849
9__inference_spectral_normalization_7_layer_call_fn_168860�
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
T__inference_spectral_normalization_7_layer_call_and_return_conditional_losses_168870
T__inference_spectral_normalization_7_layer_call_and_return_conditional_losses_168909�
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
2:0	�2spectral_normalization_7/kernel
 "
trackable_list_wrapper
-:+2spectral_normalization_7/sn_u
+:) 2spectral_normalization_4/bias
+:)2spectral_normalization_5/bias
+:)2spectral_normalization_6/bias
+:)2spectral_normalization_7/bias
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
.__inference_discriminator_layer_call_fn_167324input_2"�
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
.__inference_discriminator_layer_call_fn_167922inputs"�
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
.__inference_discriminator_layer_call_fn_167963inputs"�
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
.__inference_discriminator_layer_call_fn_167766input_2"�
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
I__inference_discriminator_layer_call_and_return_conditional_losses_168140inputs"�
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
I__inference_discriminator_layer_call_and_return_conditional_losses_168433inputs"�
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
I__inference_discriminator_layer_call_and_return_conditional_losses_167806input_2"�
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
I__inference_discriminator_layer_call_and_return_conditional_losses_167854input_2"�
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
$__inference_signature_wrapper_167889input_2"�
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
9__inference_spectral_normalization_4_layer_call_fn_168442inputs"�
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
9__inference_spectral_normalization_4_layer_call_fn_168453inputs"�
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
T__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_168464inputs"�
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
T__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_168504inputs"�
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
6__inference_layer_normalization_3_layer_call_fn_168513inputs"�
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
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_168565inputs"�
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
9__inference_spectral_normalization_5_layer_call_fn_168574inputs"�
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
9__inference_spectral_normalization_5_layer_call_fn_168585inputs"�
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
T__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_168596inputs"�
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
T__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_168636inputs"�
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
6__inference_layer_normalization_4_layer_call_fn_168645inputs"�
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
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_168697inputs"�
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
9__inference_spectral_normalization_6_layer_call_fn_168706inputs"�
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
9__inference_spectral_normalization_6_layer_call_fn_168717inputs"�
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
T__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_168728inputs"�
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
T__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_168768inputs"�
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
6__inference_layer_normalization_5_layer_call_fn_168777inputs"�
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
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_168829inputs"�
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
(__inference_flatten_layer_call_fn_168834inputs"�
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
C__inference_flatten_layer_call_and_return_conditional_losses_168840inputs"�
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
9__inference_spectral_normalization_7_layer_call_fn_168849inputs"�
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
9__inference_spectral_normalization_7_layer_call_fn_168860inputs"�
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
T__inference_spectral_normalization_7_layer_call_and_return_conditional_losses_168870inputs"�
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
T__inference_spectral_normalization_7_layer_call_and_return_conditional_losses_168909inputs"�
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
!__inference__wrapped_model_167036�[#$,\67?]IJX^8�5
.�+
)�&
input_2���������<
� "S�P
N
spectral_normalization_72�/
spectral_normalization_7����������
I__inference_discriminator_layer_call_and_return_conditional_losses_167806y[#$,\67?]IJX^@�=
6�3
)�&
input_2���������<
p 

 
� "%�"
�
0���������
� �
I__inference_discriminator_layer_call_and_return_conditional_losses_167854}[#$,.\67?A]IJXZ^@�=
6�3
)�&
input_2���������<
p

 
� "%�"
�
0���������
� �
I__inference_discriminator_layer_call_and_return_conditional_losses_168140x[#$,\67?]IJX^?�<
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
I__inference_discriminator_layer_call_and_return_conditional_losses_168433|[#$,.\67?A]IJXZ^?�<
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
.__inference_discriminator_layer_call_fn_167324l[#$,\67?]IJX^@�=
6�3
)�&
input_2���������<
p 

 
� "�����������
.__inference_discriminator_layer_call_fn_167766p[#$,.\67?A]IJXZ^@�=
6�3
)�&
input_2���������<
p

 
� "�����������
.__inference_discriminator_layer_call_fn_167922k[#$,\67?]IJX^?�<
5�2
(�%
inputs���������<
p 

 
� "�����������
.__inference_discriminator_layer_call_fn_167963o[#$,.\67?A]IJXZ^?�<
5�2
(�%
inputs���������<
p

 
� "�����������
C__inference_flatten_layer_call_and_return_conditional_losses_168840a7�4
-�*
(�%
inputs���������<
� "&�#
�
0����������
� �
(__inference_flatten_layer_call_fn_168834T7�4
-�*
(�%
inputs���������<
� "������������
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_168565l#$7�4
-�*
(�%
inputs���������< 
� "-�*
#� 
0���������< 
� �
6__inference_layer_normalization_3_layer_call_fn_168513_#$7�4
-�*
(�%
inputs���������< 
� " ����������< �
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_168697l677�4
-�*
(�%
inputs���������<
� "-�*
#� 
0���������<
� �
6__inference_layer_normalization_4_layer_call_fn_168645_677�4
-�*
(�%
inputs���������<
� " ����������<�
Q__inference_layer_normalization_5_layer_call_and_return_conditional_losses_168829lIJ7�4
-�*
(�%
inputs���������<
� "-�*
#� 
0���������<
� �
6__inference_layer_normalization_5_layer_call_fn_168777_IJ7�4
-�*
(�%
inputs���������<
� " ����������<�
$__inference_signature_wrapper_167889�[#$,\67?]IJX^C�@
� 
9�6
4
input_2)�&
input_2���������<"S�P
N
spectral_normalization_72�/
spectral_normalization_7����������
T__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_168464p[;�8
1�.
(�%
inputs���������<
p 
� "-�*
#� 
0���������< 
� �
T__inference_spectral_normalization_4_layer_call_and_return_conditional_losses_168504q[;�8
1�.
(�%
inputs���������<
p
� "-�*
#� 
0���������< 
� �
9__inference_spectral_normalization_4_layer_call_fn_168442c[;�8
1�.
(�%
inputs���������<
p 
� " ����������< �
9__inference_spectral_normalization_4_layer_call_fn_168453d[;�8
1�.
(�%
inputs���������<
p
� " ����������< �
T__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_168596p,\;�8
1�.
(�%
inputs���������< 
p 
� "-�*
#� 
0���������<
� �
T__inference_spectral_normalization_5_layer_call_and_return_conditional_losses_168636q,.\;�8
1�.
(�%
inputs���������< 
p
� "-�*
#� 
0���������<
� �
9__inference_spectral_normalization_5_layer_call_fn_168574c,\;�8
1�.
(�%
inputs���������< 
p 
� " ����������<�
9__inference_spectral_normalization_5_layer_call_fn_168585d,.\;�8
1�.
(�%
inputs���������< 
p
� " ����������<�
T__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_168728p?];�8
1�.
(�%
inputs���������<
p 
� "-�*
#� 
0���������<
� �
T__inference_spectral_normalization_6_layer_call_and_return_conditional_losses_168768q?A];�8
1�.
(�%
inputs���������<
p
� "-�*
#� 
0���������<
� �
9__inference_spectral_normalization_6_layer_call_fn_168706c?];�8
1�.
(�%
inputs���������<
p 
� " ����������<�
9__inference_spectral_normalization_6_layer_call_fn_168717d?A];�8
1�.
(�%
inputs���������<
p
� " ����������<�
T__inference_spectral_normalization_7_layer_call_and_return_conditional_losses_168870aX^4�1
*�'
!�
inputs����������
p 
� "%�"
�
0���������
� �
T__inference_spectral_normalization_7_layer_call_and_return_conditional_losses_168909bXZ^4�1
*�'
!�
inputs����������
p
� "%�"
�
0���������
� �
9__inference_spectral_normalization_7_layer_call_fn_168849TX^4�1
*�'
!�
inputs����������
p 
� "�����������
9__inference_spectral_normalization_7_layer_call_fn_168860UXZ^4�1
*�'
!�
inputs����������
p
� "����������
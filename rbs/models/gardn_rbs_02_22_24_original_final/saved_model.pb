��
��
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
p
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2	"
adj_xbool( "
adj_ybool( 
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
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
�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
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
$
DisableCopyOnRead
resource�
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
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
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
E
Relu
features"T
activations"T"
Ttype:
2	
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
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
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
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628��
�
spectral_normalization_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name spectral_normalization_17/bias
�
2spectral_normalization_17/bias/Read/ReadVariableOpReadVariableOpspectral_normalization_17/bias*
_output_shapes
:*
dtype0
�
 self_attn_model_1/conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" self_attn_model_1/conv2d_13/bias
�
4self_attn_model_1/conv2d_13/bias/Read/ReadVariableOpReadVariableOp self_attn_model_1/conv2d_13/bias*
_output_shapes
: *
dtype0
�
"self_attn_model_1/conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *3
shared_name$"self_attn_model_1/conv2d_13/kernel
�
6self_attn_model_1/conv2d_13/kernel/Read/ReadVariableOpReadVariableOp"self_attn_model_1/conv2d_13/kernel*&
_output_shapes
:  *
dtype0
�
 self_attn_model_1/conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" self_attn_model_1/conv2d_12/bias
�
4self_attn_model_1/conv2d_12/bias/Read/ReadVariableOpReadVariableOp self_attn_model_1/conv2d_12/bias*
_output_shapes
:*
dtype0
�
"self_attn_model_1/conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"self_attn_model_1/conv2d_12/kernel
�
6self_attn_model_1/conv2d_12/kernel/Read/ReadVariableOpReadVariableOp"self_attn_model_1/conv2d_12/kernel*&
_output_shapes
: *
dtype0
�
 self_attn_model_1/conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" self_attn_model_1/conv2d_11/bias
�
4self_attn_model_1/conv2d_11/bias/Read/ReadVariableOpReadVariableOp self_attn_model_1/conv2d_11/bias*
_output_shapes
:*
dtype0
�
"self_attn_model_1/conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"self_attn_model_1/conv2d_11/kernel
�
6self_attn_model_1/conv2d_11/kernel/Read/ReadVariableOpReadVariableOp"self_attn_model_1/conv2d_11/kernel*&
_output_shapes
: *
dtype0
�
Aself_attn_model_1/private__attention_1/private__attention_1_gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *R
shared_nameCAself_attn_model_1/private__attention_1/private__attention_1_gamma
�
Uself_attn_model_1/private__attention_1/private__attention_1_gamma/Read/ReadVariableOpReadVariableOpAself_attn_model_1/private__attention_1/private__attention_1_gamma*
_output_shapes
: *
dtype0
�
spectral_normalization_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name spectral_normalization_16/bias
�
2spectral_normalization_16/bias/Read/ReadVariableOpReadVariableOpspectral_normalization_16/bias*
_output_shapes
: *
dtype0
�
spectral_normalization_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name spectral_normalization_15/bias
�
2spectral_normalization_15/bias/Read/ReadVariableOpReadVariableOpspectral_normalization_15/bias*
_output_shapes
:@*
dtype0
�
spectral_normalization_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*/
shared_name spectral_normalization_14/bias
�
2spectral_normalization_14/bias/Read/ReadVariableOpReadVariableOpspectral_normalization_14/bias*
_output_shapes	
:�*
dtype0
�
spectral_normalization_17/sn_uVarHandleOp*
_output_shapes
: *
dtype0*
shape
: */
shared_name spectral_normalization_17/sn_u
�
2spectral_normalization_17/sn_u/Read/ReadVariableOpReadVariableOpspectral_normalization_17/sn_u*
_output_shapes

: *
dtype0
�
 spectral_normalization_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" spectral_normalization_17/kernel
�
4spectral_normalization_17/kernel/Read/ReadVariableOpReadVariableOp spectral_normalization_17/kernel*&
_output_shapes
: *
dtype0
�
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_5/moving_variance
�
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
: *
dtype0
�
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_5/moving_mean
�
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
: *
dtype0
�
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_5/beta
�
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
: *
dtype0
�
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_5/gamma
�
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
: *
dtype0
�
spectral_normalization_16/sn_uVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*/
shared_name spectral_normalization_16/sn_u
�
2spectral_normalization_16/sn_u/Read/ReadVariableOpReadVariableOpspectral_normalization_16/sn_u*
_output_shapes

:@*
dtype0
�
 spectral_normalization_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*1
shared_name" spectral_normalization_16/kernel
�
4spectral_normalization_16/kernel/Read/ReadVariableOpReadVariableOp spectral_normalization_16/kernel*&
_output_shapes
: @*
dtype0
�
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_4/moving_variance
�
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:@*
dtype0
�
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_4/moving_mean
�
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:@*
dtype0
�
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_4/beta
�
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:@*
dtype0
�
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_4/gamma
�
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:@*
dtype0
�
spectral_normalization_15/sn_uVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*/
shared_name spectral_normalization_15/sn_u
�
2spectral_normalization_15/sn_u/Read/ReadVariableOpReadVariableOpspectral_normalization_15/sn_u*
_output_shapes
:	�*
dtype0
�
 spectral_normalization_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*1
shared_name" spectral_normalization_15/kernel
�
4spectral_normalization_15/kernel/Read/ReadVariableOpReadVariableOp spectral_normalization_15/kernel*'
_output_shapes
:@�*
dtype0
�
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_3/moving_variance
�
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes	
:�*
dtype0
�
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_3/moving_mean
�
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_3/beta
�
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_3/gamma
�
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes	
:�*
dtype0
�
spectral_normalization_14/sn_uVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name spectral_normalization_14/sn_u
�
2spectral_normalization_14/sn_u/Read/ReadVariableOpReadVariableOpspectral_normalization_14/sn_u*
_output_shapes

:*
dtype0
�
 spectral_normalization_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*1
shared_name" spectral_normalization_14/kernel
�
4spectral_normalization_14/kernel/Read/ReadVariableOpReadVariableOp spectral_normalization_14/kernel*'
_output_shapes
:	�*
dtype0
z
serving_default_input_4Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4 spectral_normalization_14/kernelspectral_normalization_14/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variance spectral_normalization_15/kernelspectral_normalization_15/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance spectral_normalization_16/kernelspectral_normalization_16/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variance"self_attn_model_1/conv2d_11/kernel self_attn_model_1/conv2d_11/bias"self_attn_model_1/conv2d_12/kernel self_attn_model_1/conv2d_12/bias"self_attn_model_1/conv2d_13/kernel self_attn_model_1/conv2d_13/biasAself_attn_model_1/private__attention_1/private__attention_1_gamma spectral_normalization_17/kernelspectral_normalization_17/bias*'
Tin 
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*=
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_149847

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*߇
valueԇBЇ Bȇ
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
	#layer
$w
%w_shape
&sn_u
&u*
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses
-axis
	.gamma
/beta
0moving_mean
1moving_variance*
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses* 
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
Jbeta
Kmoving_mean
Lmoving_variance*
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses* 
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
	Ylayer
Zw
[w_shape
\sn_u
\u*

]	keras_api* 
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses
daxis
	egamma
fbeta
gmoving_mean
hmoving_variance*
�
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses* 
�
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses
uattn
v
query_conv
wkey_conv
x
value_conv*
�
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses
	layer
�w
�w_shape
	�sn_u
�u*
�
$0
�1
&2
.3
/4
05
16
?7
�8
A9
I10
J11
K12
L13
Z14
�15
\16
e17
f18
g19
h20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30*
�
$0
�1
.2
/3
?4
�5
I6
J7
Z8
�9
e10
f11
�12
�13
�14
�15
�16
�17
�18
�19
�20*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�serving_default* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

$0
�1
&2*

$0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

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

$kernel
	�bias
!�_jit_compiled_convolution_op*
ke
VARIABLE_VALUE spectral_normalization_14/kernel1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUE*
* 
lf
VARIABLE_VALUEspectral_normalization_14/sn_u4layer_with_weights-0/sn_u/.ATTRIBUTES/VARIABLE_VALUE*
 
.0
/1
02
13*

.0
/1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

?0
�1
A2*

?0
�1*
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

?kernel
	�bias
!�_jit_compiled_convolution_op*
ke
VARIABLE_VALUE spectral_normalization_15/kernel1layer_with_weights-2/w/.ATTRIBUTES/VARIABLE_VALUE*
* 
lf
VARIABLE_VALUEspectral_normalization_15/sn_u4layer_with_weights-2/sn_u/.ATTRIBUTES/VARIABLE_VALUE*
 
I0
J1
K2
L3*
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

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

Z0
�1
\2*

Z0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

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

Zkernel
	�bias
!�_jit_compiled_convolution_op*
ke
VARIABLE_VALUE spectral_normalization_16/kernel1layer_with_weights-4/w/.ATTRIBUTES/VARIABLE_VALUE*
* 
lf
VARIABLE_VALUEspectral_normalization_16/sn_u4layer_with_weights-4/sn_u/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
e0
f1
g2
h3*

e0
f1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_5/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_5/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_5/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_5/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
<
�0
�1
�2
�3
�4
�5
�6*
<
�0
�1
�2
�3
�4
�5
�6*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�private__attention_1_gamma

�gamma*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*

�0
�1
�2*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
ke
VARIABLE_VALUE spectral_normalization_17/kernel1layer_with_weights-7/w/.ATTRIBUTES/VARIABLE_VALUE*
* 
lf
VARIABLE_VALUEspectral_normalization_17/sn_u4layer_with_weights-7/sn_u/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEspectral_normalization_14/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEspectral_normalization_15/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEspectral_normalization_16/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAself_attn_model_1/private__attention_1/private__attention_1_gamma'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"self_attn_model_1/conv2d_11/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE self_attn_model_1/conv2d_11/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"self_attn_model_1/conv2d_12/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE self_attn_model_1/conv2d_12/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"self_attn_model_1/conv2d_13/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE self_attn_model_1/conv2d_13/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEspectral_normalization_17/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
K
&0
01
12
A3
K4
L5
\6
g7
h8
�9*
j
0
1
2
3
4
5
6
7
	8

9
10
11
12
13*
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

&0*

#0*
* 
* 
* 
* 
* 
* 
* 

$0
�1*

$0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

00
11*
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

A0*

>0*
* 
* 
* 
* 
* 
* 
* 

?0
�1*

?0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

K0
L1*
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

\0*

Y0*
* 
* 
* 
* 
* 
* 
* 

Z0
�1*

Z0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

g0
h1*
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
 
u0
v1
w2
x3*
* 
* 
* 
* 
* 
* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0*

0*
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
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

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename spectral_normalization_14/kernelspectral_normalization_14/sn_ubatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variance spectral_normalization_15/kernelspectral_normalization_15/sn_ubatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance spectral_normalization_16/kernelspectral_normalization_16/sn_ubatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variance spectral_normalization_17/kernelspectral_normalization_17/sn_uspectral_normalization_14/biasspectral_normalization_15/biasspectral_normalization_16/biasAself_attn_model_1/private__attention_1/private__attention_1_gamma"self_attn_model_1/conv2d_11/kernel self_attn_model_1/conv2d_11/bias"self_attn_model_1/conv2d_12/kernel self_attn_model_1/conv2d_12/bias"self_attn_model_1/conv2d_13/kernel self_attn_model_1/conv2d_13/biasspectral_normalization_17/biasConst*,
Tin%
#2!*
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
__inference__traced_save_151156
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename spectral_normalization_14/kernelspectral_normalization_14/sn_ubatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variance spectral_normalization_15/kernelspectral_normalization_15/sn_ubatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance spectral_normalization_16/kernelspectral_normalization_16/sn_ubatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variance spectral_normalization_17/kernelspectral_normalization_17/sn_uspectral_normalization_14/biasspectral_normalization_15/biasspectral_normalization_16/biasAself_attn_model_1/private__attention_1/private__attention_1_gamma"self_attn_model_1/conv2d_11/kernel self_attn_model_1/conv2d_11/bias"self_attn_model_1/conv2d_12/kernel self_attn_model_1/conv2d_12/bias"self_attn_model_1/conv2d_13/kernel self_attn_model_1/conv2d_13/biasspectral_normalization_17/bias*+
Tin$
"2 *
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
"__inference__traced_restore_151258��
�!
�
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_148623

inputsC
(conv2d_transpose_readvariableop_resource:@�-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
shrink_axis_mask_
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
shrink_axis_mask_
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
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
��
� 
__inference__traced_save_151156
file_prefixR
7read_disablecopyonread_spectral_normalization_14_kernel:	�I
7read_1_disablecopyonread_spectral_normalization_14_sn_u:C
4read_2_disablecopyonread_batch_normalization_3_gamma:	�B
3read_3_disablecopyonread_batch_normalization_3_beta:	�I
:read_4_disablecopyonread_batch_normalization_3_moving_mean:	�M
>read_5_disablecopyonread_batch_normalization_3_moving_variance:	�T
9read_6_disablecopyonread_spectral_normalization_15_kernel:@�J
7read_7_disablecopyonread_spectral_normalization_15_sn_u:	�B
4read_8_disablecopyonread_batch_normalization_4_gamma:@A
3read_9_disablecopyonread_batch_normalization_4_beta:@I
;read_10_disablecopyonread_batch_normalization_4_moving_mean:@M
?read_11_disablecopyonread_batch_normalization_4_moving_variance:@T
:read_12_disablecopyonread_spectral_normalization_16_kernel: @J
8read_13_disablecopyonread_spectral_normalization_16_sn_u:@C
5read_14_disablecopyonread_batch_normalization_5_gamma: B
4read_15_disablecopyonread_batch_normalization_5_beta: I
;read_16_disablecopyonread_batch_normalization_5_moving_mean: M
?read_17_disablecopyonread_batch_normalization_5_moving_variance: T
:read_18_disablecopyonread_spectral_normalization_17_kernel: J
8read_19_disablecopyonread_spectral_normalization_17_sn_u: G
8read_20_disablecopyonread_spectral_normalization_14_bias:	�F
8read_21_disablecopyonread_spectral_normalization_15_bias:@F
8read_22_disablecopyonread_spectral_normalization_16_bias: e
[read_23_disablecopyonread_self_attn_model_1_private__attention_1_private__attention_1_gamma: V
<read_24_disablecopyonread_self_attn_model_1_conv2d_11_kernel: H
:read_25_disablecopyonread_self_attn_model_1_conv2d_11_bias:V
<read_26_disablecopyonread_self_attn_model_1_conv2d_12_kernel: H
:read_27_disablecopyonread_self_attn_model_1_conv2d_12_bias:V
<read_28_disablecopyonread_self_attn_model_1_conv2d_13_kernel:  H
:read_29_disablecopyonread_self_attn_model_1_conv2d_13_bias: F
8read_30_disablecopyonread_spectral_normalization_17_bias:
savev2_const
identity_63��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
: �
Read/DisableCopyOnReadDisableCopyOnRead7read_disablecopyonread_spectral_normalization_14_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp7read_disablecopyonread_spectral_normalization_14_kernel^Read/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:	�*
dtype0r
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:	�j

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*'
_output_shapes
:	��
Read_1/DisableCopyOnReadDisableCopyOnRead7read_1_disablecopyonread_spectral_normalization_14_sn_u"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp7read_1_disablecopyonread_spectral_normalization_14_sn_u^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_2/DisableCopyOnReadDisableCopyOnRead4read_2_disablecopyonread_batch_normalization_3_gamma"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp4read_2_disablecopyonread_batch_normalization_3_gamma^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_3/DisableCopyOnReadDisableCopyOnRead3read_3_disablecopyonread_batch_normalization_3_beta"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp3read_3_disablecopyonread_batch_normalization_3_beta^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_4/DisableCopyOnReadDisableCopyOnRead:read_4_disablecopyonread_batch_normalization_3_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp:read_4_disablecopyonread_batch_normalization_3_moving_mean^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_5/DisableCopyOnReadDisableCopyOnRead>read_5_disablecopyonread_batch_normalization_3_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp>read_5_disablecopyonread_batch_normalization_3_moving_variance^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_6/DisableCopyOnReadDisableCopyOnRead9read_6_disablecopyonread_spectral_normalization_15_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp9read_6_disablecopyonread_spectral_normalization_15_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0w
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�n
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*'
_output_shapes
:@��
Read_7/DisableCopyOnReadDisableCopyOnRead7read_7_disablecopyonread_spectral_normalization_15_sn_u"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp7read_7_disablecopyonread_spectral_normalization_15_sn_u^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0o
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_8/DisableCopyOnReadDisableCopyOnRead4read_8_disablecopyonread_batch_normalization_4_gamma"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp4read_8_disablecopyonread_batch_normalization_4_gamma^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_9/DisableCopyOnReadDisableCopyOnRead3read_9_disablecopyonread_batch_normalization_4_beta"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp3read_9_disablecopyonread_batch_normalization_4_beta^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_10/DisableCopyOnReadDisableCopyOnRead;read_10_disablecopyonread_batch_normalization_4_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp;read_10_disablecopyonread_batch_normalization_4_moving_mean^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_11/DisableCopyOnReadDisableCopyOnRead?read_11_disablecopyonread_batch_normalization_4_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp?read_11_disablecopyonread_batch_normalization_4_moving_variance^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_12/DisableCopyOnReadDisableCopyOnRead:read_12_disablecopyonread_spectral_normalization_16_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp:read_12_disablecopyonread_spectral_normalization_16_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*&
_output_shapes
: @�
Read_13/DisableCopyOnReadDisableCopyOnRead8read_13_disablecopyonread_spectral_normalization_16_sn_u"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp8read_13_disablecopyonread_spectral_normalization_16_sn_u^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_14/DisableCopyOnReadDisableCopyOnRead5read_14_disablecopyonread_batch_normalization_5_gamma"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp5read_14_disablecopyonread_batch_normalization_5_gamma^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_15/DisableCopyOnReadDisableCopyOnRead4read_15_disablecopyonread_batch_normalization_5_beta"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp4read_15_disablecopyonread_batch_normalization_5_beta^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_16/DisableCopyOnReadDisableCopyOnRead;read_16_disablecopyonread_batch_normalization_5_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp;read_16_disablecopyonread_batch_normalization_5_moving_mean^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_17/DisableCopyOnReadDisableCopyOnRead?read_17_disablecopyonread_batch_normalization_5_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp?read_17_disablecopyonread_batch_normalization_5_moving_variance^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_18/DisableCopyOnReadDisableCopyOnRead:read_18_disablecopyonread_spectral_normalization_17_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp:read_18_disablecopyonread_spectral_normalization_17_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_19/DisableCopyOnReadDisableCopyOnRead8read_19_disablecopyonread_spectral_normalization_17_sn_u"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp8read_19_disablecopyonread_spectral_normalization_17_sn_u^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_20/DisableCopyOnReadDisableCopyOnRead8read_20_disablecopyonread_spectral_normalization_14_bias"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp8read_20_disablecopyonread_spectral_normalization_14_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_21/DisableCopyOnReadDisableCopyOnRead8read_21_disablecopyonread_spectral_normalization_15_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp8read_21_disablecopyonread_spectral_normalization_15_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_22/DisableCopyOnReadDisableCopyOnRead8read_22_disablecopyonread_spectral_normalization_16_bias"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp8read_22_disablecopyonread_spectral_normalization_16_bias^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_23/DisableCopyOnReadDisableCopyOnRead[read_23_disablecopyonread_self_attn_model_1_private__attention_1_private__attention_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp[read_23_disablecopyonread_self_attn_model_1_private__attention_1_private__attention_1_gamma^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_24/DisableCopyOnReadDisableCopyOnRead<read_24_disablecopyonread_self_attn_model_1_conv2d_11_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp<read_24_disablecopyonread_self_attn_model_1_conv2d_11_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_25/DisableCopyOnReadDisableCopyOnRead:read_25_disablecopyonread_self_attn_model_1_conv2d_11_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp:read_25_disablecopyonread_self_attn_model_1_conv2d_11_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_26/DisableCopyOnReadDisableCopyOnRead<read_26_disablecopyonread_self_attn_model_1_conv2d_12_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp<read_26_disablecopyonread_self_attn_model_1_conv2d_12_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_27/DisableCopyOnReadDisableCopyOnRead:read_27_disablecopyonread_self_attn_model_1_conv2d_12_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp:read_27_disablecopyonread_self_attn_model_1_conv2d_12_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_28/DisableCopyOnReadDisableCopyOnRead<read_28_disablecopyonread_self_attn_model_1_conv2d_13_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp<read_28_disablecopyonread_self_attn_model_1_conv2d_13_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0w
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  m
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*&
_output_shapes
:  �
Read_29/DisableCopyOnReadDisableCopyOnRead:read_29_disablecopyonread_self_attn_model_1_conv2d_13_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp:read_29_disablecopyonread_self_attn_model_1_conv2d_13_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_30/DisableCopyOnReadDisableCopyOnRead8read_30_disablecopyonread_spectral_normalization_17_bias"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp8read_30_disablecopyonread_spectral_normalization_17_bias^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/sn_u/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-2/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/sn_u/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-4/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/sn_u/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-7/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/sn_u/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *.
dtypes$
"2 �
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_62Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_63IdentityIdentity_62:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_63Identity_63:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:@<
:
_user_specified_name" spectral_normalization_14/kernel:>:
8
_user_specified_name spectral_normalization_14/sn_u:;7
5
_user_specified_namebatch_normalization_3/gamma::6
4
_user_specified_namebatch_normalization_3/beta:A=
;
_user_specified_name#!batch_normalization_3/moving_mean:EA
?
_user_specified_name'%batch_normalization_3/moving_variance:@<
:
_user_specified_name" spectral_normalization_15/kernel:>:
8
_user_specified_name spectral_normalization_15/sn_u:;	7
5
_user_specified_namebatch_normalization_4/gamma::
6
4
_user_specified_namebatch_normalization_4/beta:A=
;
_user_specified_name#!batch_normalization_4/moving_mean:EA
?
_user_specified_name'%batch_normalization_4/moving_variance:@<
:
_user_specified_name" spectral_normalization_16/kernel:>:
8
_user_specified_name spectral_normalization_16/sn_u:;7
5
_user_specified_namebatch_normalization_5/gamma::6
4
_user_specified_namebatch_normalization_5/beta:A=
;
_user_specified_name#!batch_normalization_5/moving_mean:EA
?
_user_specified_name'%batch_normalization_5/moving_variance:@<
:
_user_specified_name" spectral_normalization_17/kernel:>:
8
_user_specified_name spectral_normalization_17/sn_u:>:
8
_user_specified_name spectral_normalization_14/bias:>:
8
_user_specified_name spectral_normalization_15/bias:>:
8
_user_specified_name spectral_normalization_16/bias:a]
[
_user_specified_nameCAself_attn_model_1/private__attention_1/private__attention_1_gamma:B>
<
_user_specified_name$"self_attn_model_1/conv2d_11/kernel:@<
:
_user_specified_name" self_attn_model_1/conv2d_11/bias:B>
<
_user_specified_name$"self_attn_model_1/conv2d_12/kernel:@<
:
_user_specified_name" self_attn_model_1/conv2d_12/bias:B>
<
_user_specified_name$"self_attn_model_1/conv2d_13/kernel:@<
:
_user_specified_name" self_attn_model_1/conv2d_13/bias:>:
8
_user_specified_name spectral_normalization_17/bias:= 9

_output_shapes
: 

_user_specified_nameConst
�H
�
U__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_149275

inputs9
reshape_readvariableop_resource: @C
1spectral_normalize_matmul_readvariableop_resource:@@
2conv2d_transpose_6_biasadd_readvariableop_resource: 
identity��Reshape/ReadVariableOp�)conv2d_transpose_6/BiasAdd/ReadVariableOp�2conv2d_transpose_6/conv2d_transpose/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
: @*
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   s
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes

:`@�
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:@*
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

:`�
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
: @*
dtype0�
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
: @y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"          @   �
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
: @�
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(\
conv2d_transpose_6/ShapeShapeinputs*
T0*
_output_shapes
::��p
&conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_6/strided_sliceStridedSlice!conv2d_transpose_6/Shape:output:0/conv2d_transpose_6/strided_slice/stack:output:01conv2d_transpose_6/strided_slice/stack_1:output:01conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
conv2d_transpose_6/stackPack)conv2d_transpose_6/strided_slice:output:0#conv2d_transpose_6/stack/1:output:0#conv2d_transpose_6/stack/2:output:0#conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_6/strided_slice_1StridedSlice!conv2d_transpose_6/stack:output:01conv2d_transpose_6/strided_slice_1/stack:output:03conv2d_transpose_6/strided_slice_1/stack_1:output:03conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
: @*
dtype0�
#conv2d_transpose_6/conv2d_transposeConv2DBackpropInput!conv2d_transpose_6/stack:output:0:conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
)conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_transpose_6/BiasAddBiasAdd,conv2d_transpose_6/conv2d_transpose:output:01conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� z
IdentityIdentity#conv2d_transpose_6/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:��������� �
NoOpNoOp^Reshape/ReadVariableOp*^conv2d_transpose_6/BiasAdd/ReadVariableOp3^conv2d_transpose_6/conv2d_transpose/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������	@: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2V
)conv2d_transpose_6/BiasAdd/ReadVariableOp)conv2d_transpose_6/BiasAdd/ReadVariableOp2h
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:���������	@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_148650

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
E__inference_conv2d_12_layer_call_and_return_conditional_losses_150879

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
6__inference_batch_normalization_4_layer_call_fn_150141

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_148650�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:&"
 
_user_specified_name150131:&"
 
_user_specified_name150133:&"
 
_user_specified_name150135:&"
 
_user_specified_name150137
�H
�
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_149938

inputs:
reshape_readvariableop_resource:	�C
1spectral_normalize_matmul_readvariableop_resource:A
2conv2d_transpose_4_biasadd_readvariableop_resource:	�
identity��Reshape/ReadVariableOp�)conv2d_transpose_4/BiasAdd/ReadVariableOp�2conv2d_transpose_4/conv2d_transpose/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOp
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*'
_output_shapes
:	�*
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   t
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	�$�
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes
:	�$*
transpose_b(
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	�$v
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
:	�$�
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

:�
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	�$�
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
!spectral_normalize/ReadVariableOpReadVariableOpreshape_readvariableop_resource*'
_output_shapes
:	�*
dtype0�
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*'
_output_shapes
:	�y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   	   �      �
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*'
_output_shapes
:	��
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(\
conv2d_transpose_4/ShapeShapeinputs*
T0*
_output_shapes
::��p
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*'
_output_shapes
:	�*
dtype0�
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������{
IdentityIdentity#conv2d_transpose_4/BiasAdd:output:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp^Reshape/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
:__inference_spectral_normalization_17_layer_call_fn_150561

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_149554w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:&"
 
_user_specified_name150555:&"
 
_user_specified_name150557
�

�
E__inference_conv2d_12_layer_call_and_return_conditional_losses_148825

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_148754

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
_
C__inference_re_lu_4_layer_call_and_return_conditional_losses_150200

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������	@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������	@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	@:W S
/
_output_shapes
:���������	@
 
_user_specified_nameinputs
�X
�
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_150541

inputsB
(conv2d_11_conv2d_readvariableop_resource: 7
)conv2d_11_biasadd_readvariableop_resource:B
(conv2d_12_conv2d_readvariableop_resource: 7
)conv2d_12_biasadd_readvariableop_resource:B
(conv2d_13_conv2d_readvariableop_resource:  7
)conv2d_13_biasadd_readvariableop_resource: <
2private__attention_1_mul_3_readvariableop_resource: 
identity

identity_1�� conv2d_11/BiasAdd/ReadVariableOp�conv2d_11/Conv2D/ReadVariableOp� conv2d_12/BiasAdd/ReadVariableOp�conv2d_12/Conv2D/ReadVariableOp� conv2d_13/BiasAdd/ReadVariableOp�conv2d_13/Conv2D/ReadVariableOp�)private__attention_1/Mul_3/ReadVariableOp�
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_11/Conv2DConv2Dinputs'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_12/Conv2DConv2Dinputs'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
conv2d_13/Conv2DConv2Dinputs'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� r
private__attention_1/ShapeShapeconv2d_11/BiasAdd:output:0*
T0*
_output_shapes
::��r
(private__attention_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*private__attention_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*private__attention_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"private__attention_1/strided_sliceStridedSlice#private__attention_1/Shape:output:01private__attention_1/strided_slice/stack:output:03private__attention_1/strided_slice/stack_1:output:03private__attention_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
*private__attention_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,private__attention_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,private__attention_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$private__attention_1/strided_slice_1StridedSlice#private__attention_1/Shape:output:03private__attention_1/strided_slice_1/stack:output:05private__attention_1/strided_slice_1/stack_1:output:05private__attention_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
*private__attention_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,private__attention_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,private__attention_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$private__attention_1/strided_slice_2StridedSlice#private__attention_1/Shape:output:03private__attention_1/strided_slice_2/stack:output:05private__attention_1/strided_slice_2/stack_1:output:05private__attention_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
private__attention_1/mulMul-private__attention_1/strided_slice_1:output:0-private__attention_1/strided_slice_2:output:0*
T0*
_output_shapes
: o
$private__attention_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
����������
"private__attention_1/Reshape/shapePack+private__attention_1/strided_slice:output:0private__attention_1/mul:z:0-private__attention_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
private__attention_1/ReshapeReshapeconv2d_11/BiasAdd:output:0+private__attention_1/Reshape/shape:output:0*
T0*4
_output_shapes"
 :���������D����������
private__attention_1/mul_1Mul-private__attention_1/strided_slice_1:output:0-private__attention_1/strided_slice_2:output:0*
T0*
_output_shapes
: q
&private__attention_1/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
����������
$private__attention_1/Reshape_1/shapePack+private__attention_1/strided_slice:output:0private__attention_1/mul_1:z:0/private__attention_1/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
private__attention_1/Reshape_1Reshapeconv2d_12/BiasAdd:output:0-private__attention_1/Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :���������D���������x
#private__attention_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
private__attention_1/transpose	Transpose'private__attention_1/Reshape_1:output:0,private__attention_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D�
private__attention_1/mul_2Mul-private__attention_1/strided_slice_1:output:0-private__attention_1/strided_slice_2:output:0*
T0*
_output_shapes
: q
&private__attention_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
����������
$private__attention_1/Reshape_2/shapePack+private__attention_1/strided_slice:output:0private__attention_1/mul_2:z:0/private__attention_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:�
private__attention_1/Reshape_2Reshapeconv2d_13/BiasAdd:output:0-private__attention_1/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :���������D���������z
%private__attention_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
 private__attention_1/transpose_1	Transpose'private__attention_1/Reshape_2:output:0.private__attention_1/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������D�
private__attention_1/MatMulBatchMatMulV2%private__attention_1/Reshape:output:0"private__attention_1/transpose:y:0*
T0*+
_output_shapes
:���������DD�
private__attention_1/SoftmaxSoftmax$private__attention_1/MatMul:output:0*
T0*+
_output_shapes
:���������DDz
%private__attention_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
 private__attention_1/transpose_2	Transpose&private__attention_1/Softmax:softmax:0.private__attention_1/transpose_2/perm:output:0*
T0*+
_output_shapes
:���������DD�
private__attention_1/MatMul_1BatchMatMulV2$private__attention_1/transpose_1:y:0$private__attention_1/transpose_2:y:0*
T0*4
_output_shapes"
 :������������������Dz
%private__attention_1/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
 private__attention_1/transpose_3	Transpose&private__attention_1/MatMul_1:output:0.private__attention_1/transpose_3/perm:output:0*
T0*4
_output_shapes"
 :���������D���������q
&private__attention_1/Reshape_3/shape/3Const*
_output_shapes
: *
dtype0*
valueB :
����������
$private__attention_1/Reshape_3/shapePack+private__attention_1/strided_slice:output:0-private__attention_1/strided_slice_1:output:0-private__attention_1/strided_slice_2:output:0/private__attention_1/Reshape_3/shape/3:output:0*
N*
T0*
_output_shapes
:�
private__attention_1/Reshape_3Reshape$private__attention_1/transpose_3:y:0-private__attention_1/Reshape_3/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
)private__attention_1/Mul_3/ReadVariableOpReadVariableOp2private__attention_1_mul_3_readvariableop_resource*
_output_shapes
: *
dtype0�
private__attention_1/Mul_3Mul'private__attention_1/Reshape_3:output:01private__attention_1/Mul_3/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"�������������������
private__attention_1/AddAddV2private__attention_1/Mul_3:z:0inputs*
T0*/
_output_shapes
:��������� s
IdentityIdentityprivate__attention_1/Add:z:0^NoOp*
T0*/
_output_shapes
:��������� {

Identity_1Identity&private__attention_1/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:���������DD�
NoOpNoOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp*^private__attention_1/Mul_3/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):��������� : : : : : : : 2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2V
)private__attention_1/Mul_3/ReadVariableOp)private__attention_1/Mul_3/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
_
C__inference_re_lu_4_layer_call_and_return_conditional_losses_149222

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������	@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������	@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	@:W S
/
_output_shapes
:���������	@
 
_user_specified_nameinputs
�Q
�
U__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_150621

inputs9
reshape_readvariableop_resource: C
1spectral_normalize_matmul_readvariableop_resource: @
2conv2d_transpose_7_biasadd_readvariableop_resource:
identity��Reshape/ReadVariableOp�)conv2d_transpose_7/BiasAdd/ReadVariableOp�2conv2d_transpose_7/conv2d_transpose/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
: *
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����    s
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes

: �
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes

:*
transpose_b(~
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes

:v
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

:�
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

:�
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
: *
dtype0�
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
: y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
: �
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(\
conv2d_transpose_7/ShapeShapeinputs*
T0*
_output_shapes
::��p
&conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_7/strided_sliceStridedSlice!conv2d_transpose_7/Shape:output:0/conv2d_transpose_7/strided_slice/stack:output:01conv2d_transpose_7/strided_slice/stack_1:output:01conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
conv2d_transpose_7/stackPack)conv2d_transpose_7/strided_slice:output:0#conv2d_transpose_7/stack/1:output:0#conv2d_transpose_7/stack/2:output:0#conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_7/strided_slice_1StridedSlice!conv2d_transpose_7/stack:output:01conv2d_transpose_7/strided_slice_1/stack:output:03conv2d_transpose_7/strided_slice_1/stack_1:output:03conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
: *
dtype0�
#conv2d_transpose_7/conv2d_transposeConv2DBackpropInput!conv2d_transpose_7/stack:output:0:conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
)conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_transpose_7/BiasAddBiasAdd,conv2d_transpose_7/conv2d_transpose:output:01conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������]
conv2d_transpose_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>_
conv2d_transpose_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?�
conv2d_transpose_7/MulMul#conv2d_transpose_7/BiasAdd:output:0!conv2d_transpose_7/Const:output:0*
T0*/
_output_shapes
:����������
conv2d_transpose_7/AddAddV2conv2d_transpose_7/Mul:z:0#conv2d_transpose_7/Const_1:output:0*
T0*/
_output_shapes
:���������o
*conv2d_transpose_7/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
(conv2d_transpose_7/clip_by_value/MinimumMinimumconv2d_transpose_7/Add:z:03conv2d_transpose_7/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:���������g
"conv2d_transpose_7/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
 conv2d_transpose_7/clip_by_valueMaximum,conv2d_transpose_7/clip_by_value/Minimum:z:0+conv2d_transpose_7/clip_by_value/y:output:0*
T0*/
_output_shapes
:���������{
IdentityIdentity$conv2d_transpose_7/clip_by_value:z:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^Reshape/ReadVariableOp*^conv2d_transpose_7/BiasAdd/ReadVariableOp3^conv2d_transpose_7/conv2d_transpose/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:��������� : : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2V
)conv2d_transpose_7/BiasAdd/ReadVariableOp)conv2d_transpose_7/BiasAdd/ReadVariableOp2h
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
E__inference_conv2d_13_layer_call_and_return_conditional_losses_150898

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:��������� S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_150339

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
_
C__inference_re_lu_5_layer_call_and_return_conditional_losses_149300

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:��������� b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
��
�
"__inference__traced_restore_151258
file_prefixL
1assignvariableop_spectral_normalization_14_kernel:	�C
1assignvariableop_1_spectral_normalization_14_sn_u:=
.assignvariableop_2_batch_normalization_3_gamma:	�<
-assignvariableop_3_batch_normalization_3_beta:	�C
4assignvariableop_4_batch_normalization_3_moving_mean:	�G
8assignvariableop_5_batch_normalization_3_moving_variance:	�N
3assignvariableop_6_spectral_normalization_15_kernel:@�D
1assignvariableop_7_spectral_normalization_15_sn_u:	�<
.assignvariableop_8_batch_normalization_4_gamma:@;
-assignvariableop_9_batch_normalization_4_beta:@C
5assignvariableop_10_batch_normalization_4_moving_mean:@G
9assignvariableop_11_batch_normalization_4_moving_variance:@N
4assignvariableop_12_spectral_normalization_16_kernel: @D
2assignvariableop_13_spectral_normalization_16_sn_u:@=
/assignvariableop_14_batch_normalization_5_gamma: <
.assignvariableop_15_batch_normalization_5_beta: C
5assignvariableop_16_batch_normalization_5_moving_mean: G
9assignvariableop_17_batch_normalization_5_moving_variance: N
4assignvariableop_18_spectral_normalization_17_kernel: D
2assignvariableop_19_spectral_normalization_17_sn_u: A
2assignvariableop_20_spectral_normalization_14_bias:	�@
2assignvariableop_21_spectral_normalization_15_bias:@@
2assignvariableop_22_spectral_normalization_16_bias: _
Uassignvariableop_23_self_attn_model_1_private__attention_1_private__attention_1_gamma: P
6assignvariableop_24_self_attn_model_1_conv2d_11_kernel: B
4assignvariableop_25_self_attn_model_1_conv2d_11_bias:P
6assignvariableop_26_self_attn_model_1_conv2d_12_kernel: B
4assignvariableop_27_self_attn_model_1_conv2d_12_bias:P
6assignvariableop_28_self_attn_model_1_conv2d_13_kernel:  B
4assignvariableop_29_self_attn_model_1_conv2d_13_bias: @
2assignvariableop_30_spectral_normalization_17_bias:
identity_32��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/sn_u/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-2/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/sn_u/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-4/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/sn_u/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-7/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/sn_u/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::*.
dtypes$
"2 [
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp1assignvariableop_spectral_normalization_14_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp1assignvariableop_1_spectral_normalization_14_sn_uIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_3_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_3_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_3_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_3_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp3assignvariableop_6_spectral_normalization_15_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp1assignvariableop_7_spectral_normalization_15_sn_uIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_4_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_4_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_4_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_4_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp4assignvariableop_12_spectral_normalization_16_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp2assignvariableop_13_spectral_normalization_16_sn_uIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_5_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_5_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_5_moving_meanIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_5_moving_varianceIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp4assignvariableop_18_spectral_normalization_17_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp2assignvariableop_19_spectral_normalization_17_sn_uIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp2assignvariableop_20_spectral_normalization_14_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp2assignvariableop_21_spectral_normalization_15_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp2assignvariableop_22_spectral_normalization_16_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpUassignvariableop_23_self_attn_model_1_private__attention_1_private__attention_1_gammaIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp6assignvariableop_24_self_attn_model_1_conv2d_11_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp4assignvariableop_25_self_attn_model_1_conv2d_11_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp6assignvariableop_26_self_attn_model_1_conv2d_12_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp4assignvariableop_27_self_attn_model_1_conv2d_12_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp6assignvariableop_28_self_attn_model_1_conv2d_13_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp4assignvariableop_29_self_attn_model_1_conv2d_13_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp2assignvariableop_30_spectral_normalization_17_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_32Identity_32:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
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
_user_specified_namefile_prefix:@<
:
_user_specified_name" spectral_normalization_14/kernel:>:
8
_user_specified_name spectral_normalization_14/sn_u:;7
5
_user_specified_namebatch_normalization_3/gamma::6
4
_user_specified_namebatch_normalization_3/beta:A=
;
_user_specified_name#!batch_normalization_3/moving_mean:EA
?
_user_specified_name'%batch_normalization_3/moving_variance:@<
:
_user_specified_name" spectral_normalization_15/kernel:>:
8
_user_specified_name spectral_normalization_15/sn_u:;	7
5
_user_specified_namebatch_normalization_4/gamma::
6
4
_user_specified_namebatch_normalization_4/beta:A=
;
_user_specified_name#!batch_normalization_4/moving_mean:EA
?
_user_specified_name'%batch_normalization_4/moving_variance:@<
:
_user_specified_name" spectral_normalization_16/kernel:>:
8
_user_specified_name spectral_normalization_16/sn_u:;7
5
_user_specified_namebatch_normalization_5/gamma::6
4
_user_specified_namebatch_normalization_5/beta:A=
;
_user_specified_name#!batch_normalization_5/moving_mean:EA
?
_user_specified_name'%batch_normalization_5/moving_variance:@<
:
_user_specified_name" spectral_normalization_17/kernel:>:
8
_user_specified_name spectral_normalization_17/sn_u:>:
8
_user_specified_name spectral_normalization_14/bias:>:
8
_user_specified_name spectral_normalization_15/bias:>:
8
_user_specified_name spectral_normalization_16/bias:a]
[
_user_specified_nameCAself_attn_model_1/private__attention_1/private__attention_1_gamma:B>
<
_user_specified_name$"self_attn_model_1/conv2d_11/kernel:@<
:
_user_specified_name" self_attn_model_1/conv2d_11/bias:B>
<
_user_specified_name$"self_attn_model_1/conv2d_12/kernel:@<
:
_user_specified_name" self_attn_model_1/conv2d_12/bias:B>
<
_user_specified_name$"self_attn_model_1/conv2d_13/kernel:@<
:
_user_specified_name" self_attn_model_1/conv2d_13/bias:>:
8
_user_specified_name spectral_normalization_17/bias
�
�
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_148903

inputs*
conv2d_11_148811: 
conv2d_11_148813:*
conv2d_12_148826: 
conv2d_12_148828:*
conv2d_13_148841:  
conv2d_13_148843: %
private__attention_1_148897: 
identity

identity_1��!conv2d_11/StatefulPartitionedCall�!conv2d_12/StatefulPartitionedCall�!conv2d_13/StatefulPartitionedCall�,private__attention_1/StatefulPartitionedCall�
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_11_148811conv2d_11_148813*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_148810�
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_12_148826conv2d_12_148828*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_148825�
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_13_148841conv2d_13_148843*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_148840�
,private__attention_1/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*conv2d_12/StatefulPartitionedCall:output:0*conv2d_13/StatefulPartitionedCall:output:0inputsprivate__attention_1_148897*
Tin	
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:��������� :���������DD*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_private__attention_1_layer_call_and_return_conditional_losses_148896�
IdentityIdentity5private__attention_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� �

Identity_1Identity5private__attention_1/StatefulPartitionedCall:output:1^NoOp*
T0*+
_output_shapes
:���������DD�
NoOpNoOp"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall-^private__attention_1/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):��������� : : : : : : : 2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2\
,private__attention_1/StatefulPartitionedCall,private__attention_1/StatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:&"
 
_user_specified_name148811:&"
 
_user_specified_name148813:&"
 
_user_specified_name148826:&"
 
_user_specified_name148828:&"
 
_user_specified_name148841:&"
 
_user_specified_name148843:&"
 
_user_specified_name148897
�!
�
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_148727

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
shrink_axis_mask_
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
shrink_axis_mask_
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
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� ]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�!
�
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_148519

inputsC
(conv2d_transpose_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
shrink_axis_mask_
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
shrink_axis_mask_
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
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :�y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:	�*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_150357

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_150190

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
6__inference_batch_normalization_5_layer_call_fn_150308

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_148754�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:&"
 
_user_specified_name150298:&"
 
_user_specified_name150300:&"
 
_user_specified_name150302:&"
 
_user_specified_name150304
�
�
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_150128

inputsV
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource:@�@
2conv2d_transpose_5_biasadd_readvariableop_resource:@
identity��)conv2d_transpose_5/BiasAdd/ReadVariableOp�2conv2d_transpose_5/conv2d_transpose/ReadVariableOp\
conv2d_transpose_5/ShapeShapeinputs*
T0*
_output_shapes
::��p
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :	\
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:���������	@*
paddingSAME*
strides
�
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	@z
IdentityIdentity#conv2d_transpose_5/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������	@�
NoOpNoOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�H
�
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_150105

inputs:
reshape_readvariableop_resource:@�D
1spectral_normalize_matmul_readvariableop_resource:	�@
2conv2d_transpose_5_biasadd_readvariableop_resource:@
identity��Reshape/ReadVariableOp�)conv2d_transpose_5/BiasAdd/ReadVariableOp�2conv2d_transpose_5/conv2d_transpose/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOp
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*'
_output_shapes
:@�*
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   u
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0* 
_output_shapes
:
���
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes
:	�*
transpose_b(
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	�v
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
:	��
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes
:	��
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes
:	�x
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
:	��
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes
:	��
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	��
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes
:	��
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
!spectral_normalize/ReadVariableOpReadVariableOpreshape_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*'
_output_shapes
:@�y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   �   �
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*'
_output_shapes
:@��
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(\
conv2d_transpose_5/ShapeShapeinputs*
T0*
_output_shapes
::��p
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :	\
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*'
_output_shapes
:@�*
dtype0�
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:���������	@*
paddingSAME*
strides
�
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	@z
IdentityIdentity#conv2d_transpose_5/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������	@�
NoOpNoOp^Reshape/ReadVariableOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":����������: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
*__inference_conv2d_11_layer_call_fn_150850

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_148810w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:&"
 
_user_specified_name150844:&"
 
_user_specified_name150846
�
�
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_149412

inputsV
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource:	�A
2conv2d_transpose_4_biasadd_readvariableop_resource:	�
identity��)conv2d_transpose_4/BiasAdd/ReadVariableOp�2conv2d_transpose_4/conv2d_transpose/ReadVariableOp\
conv2d_transpose_4/ShapeShapeinputs*
T0*
_output_shapes
::��p
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*'
_output_shapes
:	�*
dtype0�
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������{
IdentityIdentity#conv2d_transpose_4/BiasAdd:output:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_148772

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
:__inference_spectral_normalization_14_layer_call_fn_149877

inputs"
unknown:	�
	unknown_0:
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_149127x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:&"
 
_user_specified_name149869:&"
 
_user_specified_name149871:&"
 
_user_specified_name149873
�
�
*__inference_conv2d_13_layer_call_fn_150888

inputs!
unknown:  
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
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_148840w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:&"
 
_user_specified_name150882:&"
 
_user_specified_name150884
�
a
E__inference_reshape_1_layer_call_and_return_conditional_losses_149866

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�!
!__inference__wrapped_model_148486
input_4x
]model_1_spectral_normalization_14_conv2d_transpose_4_conv2d_transpose_readvariableop_resource:	�c
Tmodel_1_spectral_normalization_14_conv2d_transpose_4_biasadd_readvariableop_resource:	�D
5model_1_batch_normalization_3_readvariableop_resource:	�F
7model_1_batch_normalization_3_readvariableop_1_resource:	�U
Fmodel_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	�W
Hmodel_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	�x
]model_1_spectral_normalization_15_conv2d_transpose_5_conv2d_transpose_readvariableop_resource:@�b
Tmodel_1_spectral_normalization_15_conv2d_transpose_5_biasadd_readvariableop_resource:@C
5model_1_batch_normalization_4_readvariableop_resource:@E
7model_1_batch_normalization_4_readvariableop_1_resource:@T
Fmodel_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:@V
Hmodel_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:@w
]model_1_spectral_normalization_16_conv2d_transpose_6_conv2d_transpose_readvariableop_resource: @b
Tmodel_1_spectral_normalization_16_conv2d_transpose_6_biasadd_readvariableop_resource: C
5model_1_batch_normalization_5_readvariableop_resource: E
7model_1_batch_normalization_5_readvariableop_1_resource: T
Fmodel_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource: V
Hmodel_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource: \
Bmodel_1_self_attn_model_1_conv2d_11_conv2d_readvariableop_resource: Q
Cmodel_1_self_attn_model_1_conv2d_11_biasadd_readvariableop_resource:\
Bmodel_1_self_attn_model_1_conv2d_12_conv2d_readvariableop_resource: Q
Cmodel_1_self_attn_model_1_conv2d_12_biasadd_readvariableop_resource:\
Bmodel_1_self_attn_model_1_conv2d_13_conv2d_readvariableop_resource:  Q
Cmodel_1_self_attn_model_1_conv2d_13_biasadd_readvariableop_resource: V
Lmodel_1_self_attn_model_1_private__attention_1_mul_3_readvariableop_resource: w
]model_1_spectral_normalization_17_conv2d_transpose_7_conv2d_transpose_readvariableop_resource: b
Tmodel_1_spectral_normalization_17_conv2d_transpose_7_biasadd_readvariableop_resource:
identity��=model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�?model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�,model_1/batch_normalization_3/ReadVariableOp�.model_1/batch_normalization_3/ReadVariableOp_1�=model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp�?model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�,model_1/batch_normalization_4/ReadVariableOp�.model_1/batch_normalization_4/ReadVariableOp_1�=model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp�?model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�,model_1/batch_normalization_5/ReadVariableOp�.model_1/batch_normalization_5/ReadVariableOp_1�:model_1/self_attn_model_1/conv2d_11/BiasAdd/ReadVariableOp�9model_1/self_attn_model_1/conv2d_11/Conv2D/ReadVariableOp�:model_1/self_attn_model_1/conv2d_12/BiasAdd/ReadVariableOp�9model_1/self_attn_model_1/conv2d_12/Conv2D/ReadVariableOp�:model_1/self_attn_model_1/conv2d_13/BiasAdd/ReadVariableOp�9model_1/self_attn_model_1/conv2d_13/Conv2D/ReadVariableOp�Cmodel_1/self_attn_model_1/private__attention_1/Mul_3/ReadVariableOp�Kmodel_1/spectral_normalization_14/conv2d_transpose_4/BiasAdd/ReadVariableOp�Tmodel_1/spectral_normalization_14/conv2d_transpose_4/conv2d_transpose/ReadVariableOp�Kmodel_1/spectral_normalization_15/conv2d_transpose_5/BiasAdd/ReadVariableOp�Tmodel_1/spectral_normalization_15/conv2d_transpose_5/conv2d_transpose/ReadVariableOp�Kmodel_1/spectral_normalization_16/conv2d_transpose_6/BiasAdd/ReadVariableOp�Tmodel_1/spectral_normalization_16/conv2d_transpose_6/conv2d_transpose/ReadVariableOp�Kmodel_1/spectral_normalization_17/conv2d_transpose_7/BiasAdd/ReadVariableOp�Tmodel_1/spectral_normalization_17/conv2d_transpose_7/conv2d_transpose/ReadVariableOp\
model_1/reshape_1/ShapeShapeinput_4*
T0*
_output_shapes
::��o
%model_1/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'model_1/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'model_1/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model_1/reshape_1/strided_sliceStridedSlice model_1/reshape_1/Shape:output:0.model_1/reshape_1/strided_slice/stack:output:00model_1/reshape_1/strided_slice/stack_1:output:00model_1/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model_1/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :c
!model_1/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :c
!model_1/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
model_1/reshape_1/Reshape/shapePack(model_1/reshape_1/strided_slice:output:0*model_1/reshape_1/Reshape/shape/1:output:0*model_1/reshape_1/Reshape/shape/2:output:0*model_1/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
model_1/reshape_1/ReshapeReshapeinput_4(model_1/reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
:model_1/spectral_normalization_14/conv2d_transpose_4/ShapeShape"model_1/reshape_1/Reshape:output:0*
T0*
_output_shapes
::���
Hmodel_1/spectral_normalization_14/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Jmodel_1/spectral_normalization_14/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Jmodel_1/spectral_normalization_14/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Bmodel_1/spectral_normalization_14/conv2d_transpose_4/strided_sliceStridedSliceCmodel_1/spectral_normalization_14/conv2d_transpose_4/Shape:output:0Qmodel_1/spectral_normalization_14/conv2d_transpose_4/strided_slice/stack:output:0Smodel_1/spectral_normalization_14/conv2d_transpose_4/strided_slice/stack_1:output:0Smodel_1/spectral_normalization_14/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
<model_1/spectral_normalization_14/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :~
<model_1/spectral_normalization_14/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :
<model_1/spectral_normalization_14/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
:model_1/spectral_normalization_14/conv2d_transpose_4/stackPackKmodel_1/spectral_normalization_14/conv2d_transpose_4/strided_slice:output:0Emodel_1/spectral_normalization_14/conv2d_transpose_4/stack/1:output:0Emodel_1/spectral_normalization_14/conv2d_transpose_4/stack/2:output:0Emodel_1/spectral_normalization_14/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:�
Jmodel_1/spectral_normalization_14/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Lmodel_1/spectral_normalization_14/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Lmodel_1/spectral_normalization_14/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Dmodel_1/spectral_normalization_14/conv2d_transpose_4/strided_slice_1StridedSliceCmodel_1/spectral_normalization_14/conv2d_transpose_4/stack:output:0Smodel_1/spectral_normalization_14/conv2d_transpose_4/strided_slice_1/stack:output:0Umodel_1/spectral_normalization_14/conv2d_transpose_4/strided_slice_1/stack_1:output:0Umodel_1/spectral_normalization_14/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Tmodel_1/spectral_normalization_14/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp]model_1_spectral_normalization_14_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*'
_output_shapes
:	�*
dtype0�
Emodel_1/spectral_normalization_14/conv2d_transpose_4/conv2d_transposeConv2DBackpropInputCmodel_1/spectral_normalization_14/conv2d_transpose_4/stack:output:0\model_1/spectral_normalization_14/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0"model_1/reshape_1/Reshape:output:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
Kmodel_1/spectral_normalization_14/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOpTmodel_1_spectral_normalization_14_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<model_1/spectral_normalization_14/conv2d_transpose_4/BiasAddBiasAddNmodel_1/spectral_normalization_14/conv2d_transpose_4/conv2d_transpose:output:0Smodel_1/spectral_normalization_14/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
,model_1/batch_normalization_3/ReadVariableOpReadVariableOp5model_1_batch_normalization_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.model_1/batch_normalization_3/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
=model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
?model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
.model_1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3Emodel_1/spectral_normalization_14/conv2d_transpose_4/BiasAdd:output:04model_1/batch_normalization_3/ReadVariableOp:value:06model_1/batch_normalization_3/ReadVariableOp_1:value:0Emodel_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
model_1/re_lu_3/ReluRelu2model_1/batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
:model_1/spectral_normalization_15/conv2d_transpose_5/ShapeShape"model_1/re_lu_3/Relu:activations:0*
T0*
_output_shapes
::���
Hmodel_1/spectral_normalization_15/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Jmodel_1/spectral_normalization_15/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Jmodel_1/spectral_normalization_15/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Bmodel_1/spectral_normalization_15/conv2d_transpose_5/strided_sliceStridedSliceCmodel_1/spectral_normalization_15/conv2d_transpose_5/Shape:output:0Qmodel_1/spectral_normalization_15/conv2d_transpose_5/strided_slice/stack:output:0Smodel_1/spectral_normalization_15/conv2d_transpose_5/strided_slice/stack_1:output:0Smodel_1/spectral_normalization_15/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
<model_1/spectral_normalization_15/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :~
<model_1/spectral_normalization_15/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :	~
<model_1/spectral_normalization_15/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
:model_1/spectral_normalization_15/conv2d_transpose_5/stackPackKmodel_1/spectral_normalization_15/conv2d_transpose_5/strided_slice:output:0Emodel_1/spectral_normalization_15/conv2d_transpose_5/stack/1:output:0Emodel_1/spectral_normalization_15/conv2d_transpose_5/stack/2:output:0Emodel_1/spectral_normalization_15/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:�
Jmodel_1/spectral_normalization_15/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Lmodel_1/spectral_normalization_15/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Lmodel_1/spectral_normalization_15/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Dmodel_1/spectral_normalization_15/conv2d_transpose_5/strided_slice_1StridedSliceCmodel_1/spectral_normalization_15/conv2d_transpose_5/stack:output:0Smodel_1/spectral_normalization_15/conv2d_transpose_5/strided_slice_1/stack:output:0Umodel_1/spectral_normalization_15/conv2d_transpose_5/strided_slice_1/stack_1:output:0Umodel_1/spectral_normalization_15/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Tmodel_1/spectral_normalization_15/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp]model_1_spectral_normalization_15_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Emodel_1/spectral_normalization_15/conv2d_transpose_5/conv2d_transposeConv2DBackpropInputCmodel_1/spectral_normalization_15/conv2d_transpose_5/stack:output:0\model_1/spectral_normalization_15/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0"model_1/re_lu_3/Relu:activations:0*
T0*/
_output_shapes
:���������	@*
paddingSAME*
strides
�
Kmodel_1/spectral_normalization_15/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOpTmodel_1_spectral_normalization_15_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
<model_1/spectral_normalization_15/conv2d_transpose_5/BiasAddBiasAddNmodel_1/spectral_normalization_15/conv2d_transpose_5/conv2d_transpose:output:0Smodel_1/spectral_normalization_15/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	@�
,model_1/batch_normalization_4/ReadVariableOpReadVariableOp5model_1_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype0�
.model_1/batch_normalization_4/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
=model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
?model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
.model_1/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3Emodel_1/spectral_normalization_15/conv2d_transpose_5/BiasAdd:output:04model_1/batch_normalization_4/ReadVariableOp:value:06model_1/batch_normalization_4/ReadVariableOp_1:value:0Emodel_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������	@:@:@:@:@:*
epsilon%o�:*
is_training( �
model_1/re_lu_4/ReluRelu2model_1/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������	@�
:model_1/spectral_normalization_16/conv2d_transpose_6/ShapeShape"model_1/re_lu_4/Relu:activations:0*
T0*
_output_shapes
::���
Hmodel_1/spectral_normalization_16/conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Jmodel_1/spectral_normalization_16/conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Jmodel_1/spectral_normalization_16/conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Bmodel_1/spectral_normalization_16/conv2d_transpose_6/strided_sliceStridedSliceCmodel_1/spectral_normalization_16/conv2d_transpose_6/Shape:output:0Qmodel_1/spectral_normalization_16/conv2d_transpose_6/strided_slice/stack:output:0Smodel_1/spectral_normalization_16/conv2d_transpose_6/strided_slice/stack_1:output:0Smodel_1/spectral_normalization_16/conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
<model_1/spectral_normalization_16/conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :~
<model_1/spectral_normalization_16/conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :~
<model_1/spectral_normalization_16/conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
:model_1/spectral_normalization_16/conv2d_transpose_6/stackPackKmodel_1/spectral_normalization_16/conv2d_transpose_6/strided_slice:output:0Emodel_1/spectral_normalization_16/conv2d_transpose_6/stack/1:output:0Emodel_1/spectral_normalization_16/conv2d_transpose_6/stack/2:output:0Emodel_1/spectral_normalization_16/conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:�
Jmodel_1/spectral_normalization_16/conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Lmodel_1/spectral_normalization_16/conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Lmodel_1/spectral_normalization_16/conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Dmodel_1/spectral_normalization_16/conv2d_transpose_6/strided_slice_1StridedSliceCmodel_1/spectral_normalization_16/conv2d_transpose_6/stack:output:0Smodel_1/spectral_normalization_16/conv2d_transpose_6/strided_slice_1/stack:output:0Umodel_1/spectral_normalization_16/conv2d_transpose_6/strided_slice_1/stack_1:output:0Umodel_1/spectral_normalization_16/conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Tmodel_1/spectral_normalization_16/conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOp]model_1_spectral_normalization_16_conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Emodel_1/spectral_normalization_16/conv2d_transpose_6/conv2d_transposeConv2DBackpropInputCmodel_1/spectral_normalization_16/conv2d_transpose_6/stack:output:0\model_1/spectral_normalization_16/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0"model_1/re_lu_4/Relu:activations:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
Kmodel_1/spectral_normalization_16/conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOpTmodel_1_spectral_normalization_16_conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
<model_1/spectral_normalization_16/conv2d_transpose_6/BiasAddBiasAddNmodel_1/spectral_normalization_16/conv2d_transpose_6/conv2d_transpose:output:0Smodel_1/spectral_normalization_16/conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
6model_1/tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                �
8model_1/tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               �
8model_1/tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            �
0model_1/tf.__operators__.getitem_1/strided_sliceStridedSliceEmodel_1/spectral_normalization_16/conv2d_transpose_6/BiasAdd:output:0?model_1/tf.__operators__.getitem_1/strided_slice/stack:output:0Amodel_1/tf.__operators__.getitem_1/strided_slice/stack_1:output:0Amodel_1/tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:��������� *

begin_mask*
end_mask�
,model_1/batch_normalization_5/ReadVariableOpReadVariableOp5model_1_batch_normalization_5_readvariableop_resource*
_output_shapes
: *
dtype0�
.model_1/batch_normalization_5/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_5_readvariableop_1_resource*
_output_shapes
: *
dtype0�
=model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
?model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
.model_1/batch_normalization_5/FusedBatchNormV3FusedBatchNormV39model_1/tf.__operators__.getitem_1/strided_slice:output:04model_1/batch_normalization_5/ReadVariableOp:value:06model_1/batch_normalization_5/ReadVariableOp_1:value:0Emodel_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:��������� : : : : :*
epsilon%o�:*
is_training( �
model_1/re_lu_5/ReluRelu2model_1/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:��������� �
9model_1/self_attn_model_1/conv2d_11/Conv2D/ReadVariableOpReadVariableOpBmodel_1_self_attn_model_1_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
*model_1/self_attn_model_1/conv2d_11/Conv2DConv2D"model_1/re_lu_5/Relu:activations:0Amodel_1/self_attn_model_1/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
:model_1/self_attn_model_1/conv2d_11/BiasAdd/ReadVariableOpReadVariableOpCmodel_1_self_attn_model_1_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
+model_1/self_attn_model_1/conv2d_11/BiasAddBiasAdd3model_1/self_attn_model_1/conv2d_11/Conv2D:output:0Bmodel_1/self_attn_model_1/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
9model_1/self_attn_model_1/conv2d_12/Conv2D/ReadVariableOpReadVariableOpBmodel_1_self_attn_model_1_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
*model_1/self_attn_model_1/conv2d_12/Conv2DConv2D"model_1/re_lu_5/Relu:activations:0Amodel_1/self_attn_model_1/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
:model_1/self_attn_model_1/conv2d_12/BiasAdd/ReadVariableOpReadVariableOpCmodel_1_self_attn_model_1_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
+model_1/self_attn_model_1/conv2d_12/BiasAddBiasAdd3model_1/self_attn_model_1/conv2d_12/Conv2D:output:0Bmodel_1/self_attn_model_1/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
9model_1/self_attn_model_1/conv2d_13/Conv2D/ReadVariableOpReadVariableOpBmodel_1_self_attn_model_1_conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
*model_1/self_attn_model_1/conv2d_13/Conv2DConv2D"model_1/re_lu_5/Relu:activations:0Amodel_1/self_attn_model_1/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
:model_1/self_attn_model_1/conv2d_13/BiasAdd/ReadVariableOpReadVariableOpCmodel_1_self_attn_model_1_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
+model_1/self_attn_model_1/conv2d_13/BiasAddBiasAdd3model_1/self_attn_model_1/conv2d_13/Conv2D:output:0Bmodel_1/self_attn_model_1/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
4model_1/self_attn_model_1/private__attention_1/ShapeShape4model_1/self_attn_model_1/conv2d_11/BiasAdd:output:0*
T0*
_output_shapes
::���
Bmodel_1/self_attn_model_1/private__attention_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Dmodel_1/self_attn_model_1/private__attention_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Dmodel_1/self_attn_model_1/private__attention_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
<model_1/self_attn_model_1/private__attention_1/strided_sliceStridedSlice=model_1/self_attn_model_1/private__attention_1/Shape:output:0Kmodel_1/self_attn_model_1/private__attention_1/strided_slice/stack:output:0Mmodel_1/self_attn_model_1/private__attention_1/strided_slice/stack_1:output:0Mmodel_1/self_attn_model_1/private__attention_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Dmodel_1/self_attn_model_1/private__attention_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Fmodel_1/self_attn_model_1/private__attention_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Fmodel_1/self_attn_model_1/private__attention_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
>model_1/self_attn_model_1/private__attention_1/strided_slice_1StridedSlice=model_1/self_attn_model_1/private__attention_1/Shape:output:0Mmodel_1/self_attn_model_1/private__attention_1/strided_slice_1/stack:output:0Omodel_1/self_attn_model_1/private__attention_1/strided_slice_1/stack_1:output:0Omodel_1/self_attn_model_1/private__attention_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Dmodel_1/self_attn_model_1/private__attention_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Fmodel_1/self_attn_model_1/private__attention_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Fmodel_1/self_attn_model_1/private__attention_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
>model_1/self_attn_model_1/private__attention_1/strided_slice_2StridedSlice=model_1/self_attn_model_1/private__attention_1/Shape:output:0Mmodel_1/self_attn_model_1/private__attention_1/strided_slice_2/stack:output:0Omodel_1/self_attn_model_1/private__attention_1/strided_slice_2/stack_1:output:0Omodel_1/self_attn_model_1/private__attention_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2model_1/self_attn_model_1/private__attention_1/mulMulGmodel_1/self_attn_model_1/private__attention_1/strided_slice_1:output:0Gmodel_1/self_attn_model_1/private__attention_1/strided_slice_2:output:0*
T0*
_output_shapes
: �
>model_1/self_attn_model_1/private__attention_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
����������
<model_1/self_attn_model_1/private__attention_1/Reshape/shapePackEmodel_1/self_attn_model_1/private__attention_1/strided_slice:output:06model_1/self_attn_model_1/private__attention_1/mul:z:0Gmodel_1/self_attn_model_1/private__attention_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
6model_1/self_attn_model_1/private__attention_1/ReshapeReshape4model_1/self_attn_model_1/conv2d_11/BiasAdd:output:0Emodel_1/self_attn_model_1/private__attention_1/Reshape/shape:output:0*
T0*4
_output_shapes"
 :���������D����������
4model_1/self_attn_model_1/private__attention_1/mul_1MulGmodel_1/self_attn_model_1/private__attention_1/strided_slice_1:output:0Gmodel_1/self_attn_model_1/private__attention_1/strided_slice_2:output:0*
T0*
_output_shapes
: �
@model_1/self_attn_model_1/private__attention_1/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
����������
>model_1/self_attn_model_1/private__attention_1/Reshape_1/shapePackEmodel_1/self_attn_model_1/private__attention_1/strided_slice:output:08model_1/self_attn_model_1/private__attention_1/mul_1:z:0Imodel_1/self_attn_model_1/private__attention_1/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
8model_1/self_attn_model_1/private__attention_1/Reshape_1Reshape4model_1/self_attn_model_1/conv2d_12/BiasAdd:output:0Gmodel_1/self_attn_model_1/private__attention_1/Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :���������D����������
=model_1/self_attn_model_1/private__attention_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
8model_1/self_attn_model_1/private__attention_1/transpose	TransposeAmodel_1/self_attn_model_1/private__attention_1/Reshape_1:output:0Fmodel_1/self_attn_model_1/private__attention_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D�
4model_1/self_attn_model_1/private__attention_1/mul_2MulGmodel_1/self_attn_model_1/private__attention_1/strided_slice_1:output:0Gmodel_1/self_attn_model_1/private__attention_1/strided_slice_2:output:0*
T0*
_output_shapes
: �
@model_1/self_attn_model_1/private__attention_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
����������
>model_1/self_attn_model_1/private__attention_1/Reshape_2/shapePackEmodel_1/self_attn_model_1/private__attention_1/strided_slice:output:08model_1/self_attn_model_1/private__attention_1/mul_2:z:0Imodel_1/self_attn_model_1/private__attention_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:�
8model_1/self_attn_model_1/private__attention_1/Reshape_2Reshape4model_1/self_attn_model_1/conv2d_13/BiasAdd:output:0Gmodel_1/self_attn_model_1/private__attention_1/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :���������D����������
?model_1/self_attn_model_1/private__attention_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
:model_1/self_attn_model_1/private__attention_1/transpose_1	TransposeAmodel_1/self_attn_model_1/private__attention_1/Reshape_2:output:0Hmodel_1/self_attn_model_1/private__attention_1/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������D�
5model_1/self_attn_model_1/private__attention_1/MatMulBatchMatMulV2?model_1/self_attn_model_1/private__attention_1/Reshape:output:0<model_1/self_attn_model_1/private__attention_1/transpose:y:0*
T0*+
_output_shapes
:���������DD�
6model_1/self_attn_model_1/private__attention_1/SoftmaxSoftmax>model_1/self_attn_model_1/private__attention_1/MatMul:output:0*
T0*+
_output_shapes
:���������DD�
?model_1/self_attn_model_1/private__attention_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
:model_1/self_attn_model_1/private__attention_1/transpose_2	Transpose@model_1/self_attn_model_1/private__attention_1/Softmax:softmax:0Hmodel_1/self_attn_model_1/private__attention_1/transpose_2/perm:output:0*
T0*+
_output_shapes
:���������DD�
7model_1/self_attn_model_1/private__attention_1/MatMul_1BatchMatMulV2>model_1/self_attn_model_1/private__attention_1/transpose_1:y:0>model_1/self_attn_model_1/private__attention_1/transpose_2:y:0*
T0*4
_output_shapes"
 :������������������D�
?model_1/self_attn_model_1/private__attention_1/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
:model_1/self_attn_model_1/private__attention_1/transpose_3	Transpose@model_1/self_attn_model_1/private__attention_1/MatMul_1:output:0Hmodel_1/self_attn_model_1/private__attention_1/transpose_3/perm:output:0*
T0*4
_output_shapes"
 :���������D����������
@model_1/self_attn_model_1/private__attention_1/Reshape_3/shape/3Const*
_output_shapes
: *
dtype0*
valueB :
����������
>model_1/self_attn_model_1/private__attention_1/Reshape_3/shapePackEmodel_1/self_attn_model_1/private__attention_1/strided_slice:output:0Gmodel_1/self_attn_model_1/private__attention_1/strided_slice_1:output:0Gmodel_1/self_attn_model_1/private__attention_1/strided_slice_2:output:0Imodel_1/self_attn_model_1/private__attention_1/Reshape_3/shape/3:output:0*
N*
T0*
_output_shapes
:�
8model_1/self_attn_model_1/private__attention_1/Reshape_3Reshape>model_1/self_attn_model_1/private__attention_1/transpose_3:y:0Gmodel_1/self_attn_model_1/private__attention_1/Reshape_3/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
Cmodel_1/self_attn_model_1/private__attention_1/Mul_3/ReadVariableOpReadVariableOpLmodel_1_self_attn_model_1_private__attention_1_mul_3_readvariableop_resource*
_output_shapes
: *
dtype0�
4model_1/self_attn_model_1/private__attention_1/Mul_3MulAmodel_1/self_attn_model_1/private__attention_1/Reshape_3:output:0Kmodel_1/self_attn_model_1/private__attention_1/Mul_3/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"�������������������
2model_1/self_attn_model_1/private__attention_1/AddAddV28model_1/self_attn_model_1/private__attention_1/Mul_3:z:0"model_1/re_lu_5/Relu:activations:0*
T0*/
_output_shapes
:��������� �
:model_1/spectral_normalization_17/conv2d_transpose_7/ShapeShape6model_1/self_attn_model_1/private__attention_1/Add:z:0*
T0*
_output_shapes
::���
Hmodel_1/spectral_normalization_17/conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Jmodel_1/spectral_normalization_17/conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Jmodel_1/spectral_normalization_17/conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Bmodel_1/spectral_normalization_17/conv2d_transpose_7/strided_sliceStridedSliceCmodel_1/spectral_normalization_17/conv2d_transpose_7/Shape:output:0Qmodel_1/spectral_normalization_17/conv2d_transpose_7/strided_slice/stack:output:0Smodel_1/spectral_normalization_17/conv2d_transpose_7/strided_slice/stack_1:output:0Smodel_1/spectral_normalization_17/conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
<model_1/spectral_normalization_17/conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :~
<model_1/spectral_normalization_17/conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :~
<model_1/spectral_normalization_17/conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
:model_1/spectral_normalization_17/conv2d_transpose_7/stackPackKmodel_1/spectral_normalization_17/conv2d_transpose_7/strided_slice:output:0Emodel_1/spectral_normalization_17/conv2d_transpose_7/stack/1:output:0Emodel_1/spectral_normalization_17/conv2d_transpose_7/stack/2:output:0Emodel_1/spectral_normalization_17/conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:�
Jmodel_1/spectral_normalization_17/conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Lmodel_1/spectral_normalization_17/conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Lmodel_1/spectral_normalization_17/conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Dmodel_1/spectral_normalization_17/conv2d_transpose_7/strided_slice_1StridedSliceCmodel_1/spectral_normalization_17/conv2d_transpose_7/stack:output:0Smodel_1/spectral_normalization_17/conv2d_transpose_7/strided_slice_1/stack:output:0Umodel_1/spectral_normalization_17/conv2d_transpose_7/strided_slice_1/stack_1:output:0Umodel_1/spectral_normalization_17/conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Tmodel_1/spectral_normalization_17/conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOp]model_1_spectral_normalization_17_conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
Emodel_1/spectral_normalization_17/conv2d_transpose_7/conv2d_transposeConv2DBackpropInputCmodel_1/spectral_normalization_17/conv2d_transpose_7/stack:output:0\model_1/spectral_normalization_17/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:06model_1/self_attn_model_1/private__attention_1/Add:z:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
Kmodel_1/spectral_normalization_17/conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOpTmodel_1_spectral_normalization_17_conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
<model_1/spectral_normalization_17/conv2d_transpose_7/BiasAddBiasAddNmodel_1/spectral_normalization_17/conv2d_transpose_7/conv2d_transpose:output:0Smodel_1/spectral_normalization_17/conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������
:model_1/spectral_normalization_17/conv2d_transpose_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
<model_1/spectral_normalization_17/conv2d_transpose_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?�
8model_1/spectral_normalization_17/conv2d_transpose_7/MulMulEmodel_1/spectral_normalization_17/conv2d_transpose_7/BiasAdd:output:0Cmodel_1/spectral_normalization_17/conv2d_transpose_7/Const:output:0*
T0*/
_output_shapes
:����������
8model_1/spectral_normalization_17/conv2d_transpose_7/AddAddV2<model_1/spectral_normalization_17/conv2d_transpose_7/Mul:z:0Emodel_1/spectral_normalization_17/conv2d_transpose_7/Const_1:output:0*
T0*/
_output_shapes
:����������
Lmodel_1/spectral_normalization_17/conv2d_transpose_7/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Jmodel_1/spectral_normalization_17/conv2d_transpose_7/clip_by_value/MinimumMinimum<model_1/spectral_normalization_17/conv2d_transpose_7/Add:z:0Umodel_1/spectral_normalization_17/conv2d_transpose_7/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:����������
Dmodel_1/spectral_normalization_17/conv2d_transpose_7/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
Bmodel_1/spectral_normalization_17/conv2d_transpose_7/clip_by_valueMaximumNmodel_1/spectral_normalization_17/conv2d_transpose_7/clip_by_value/Minimum:z:0Mmodel_1/spectral_normalization_17/conv2d_transpose_7/clip_by_value/y:output:0*
T0*/
_output_shapes
:����������
IdentityIdentityFmodel_1/spectral_normalization_17/conv2d_transpose_7/clip_by_value:z:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp>^model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp@^model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1-^model_1/batch_normalization_3/ReadVariableOp/^model_1/batch_normalization_3/ReadVariableOp_1>^model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp@^model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1-^model_1/batch_normalization_4/ReadVariableOp/^model_1/batch_normalization_4/ReadVariableOp_1>^model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp@^model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1-^model_1/batch_normalization_5/ReadVariableOp/^model_1/batch_normalization_5/ReadVariableOp_1;^model_1/self_attn_model_1/conv2d_11/BiasAdd/ReadVariableOp:^model_1/self_attn_model_1/conv2d_11/Conv2D/ReadVariableOp;^model_1/self_attn_model_1/conv2d_12/BiasAdd/ReadVariableOp:^model_1/self_attn_model_1/conv2d_12/Conv2D/ReadVariableOp;^model_1/self_attn_model_1/conv2d_13/BiasAdd/ReadVariableOp:^model_1/self_attn_model_1/conv2d_13/Conv2D/ReadVariableOpD^model_1/self_attn_model_1/private__attention_1/Mul_3/ReadVariableOpL^model_1/spectral_normalization_14/conv2d_transpose_4/BiasAdd/ReadVariableOpU^model_1/spectral_normalization_14/conv2d_transpose_4/conv2d_transpose/ReadVariableOpL^model_1/spectral_normalization_15/conv2d_transpose_5/BiasAdd/ReadVariableOpU^model_1/spectral_normalization_15/conv2d_transpose_5/conv2d_transpose/ReadVariableOpL^model_1/spectral_normalization_16/conv2d_transpose_6/BiasAdd/ReadVariableOpU^model_1/spectral_normalization_16/conv2d_transpose_6/conv2d_transpose/ReadVariableOpL^model_1/spectral_normalization_17/conv2d_transpose_7/BiasAdd/ReadVariableOpU^model_1/spectral_normalization_17/conv2d_transpose_7/conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : 2~
=model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp=model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2�
?model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12\
,model_1/batch_normalization_3/ReadVariableOp,model_1/batch_normalization_3/ReadVariableOp2`
.model_1/batch_normalization_3/ReadVariableOp_1.model_1/batch_normalization_3/ReadVariableOp_12~
=model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp=model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2�
?model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12\
,model_1/batch_normalization_4/ReadVariableOp,model_1/batch_normalization_4/ReadVariableOp2`
.model_1/batch_normalization_4/ReadVariableOp_1.model_1/batch_normalization_4/ReadVariableOp_12~
=model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp=model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2�
?model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12\
,model_1/batch_normalization_5/ReadVariableOp,model_1/batch_normalization_5/ReadVariableOp2`
.model_1/batch_normalization_5/ReadVariableOp_1.model_1/batch_normalization_5/ReadVariableOp_12x
:model_1/self_attn_model_1/conv2d_11/BiasAdd/ReadVariableOp:model_1/self_attn_model_1/conv2d_11/BiasAdd/ReadVariableOp2v
9model_1/self_attn_model_1/conv2d_11/Conv2D/ReadVariableOp9model_1/self_attn_model_1/conv2d_11/Conv2D/ReadVariableOp2x
:model_1/self_attn_model_1/conv2d_12/BiasAdd/ReadVariableOp:model_1/self_attn_model_1/conv2d_12/BiasAdd/ReadVariableOp2v
9model_1/self_attn_model_1/conv2d_12/Conv2D/ReadVariableOp9model_1/self_attn_model_1/conv2d_12/Conv2D/ReadVariableOp2x
:model_1/self_attn_model_1/conv2d_13/BiasAdd/ReadVariableOp:model_1/self_attn_model_1/conv2d_13/BiasAdd/ReadVariableOp2v
9model_1/self_attn_model_1/conv2d_13/Conv2D/ReadVariableOp9model_1/self_attn_model_1/conv2d_13/Conv2D/ReadVariableOp2�
Cmodel_1/self_attn_model_1/private__attention_1/Mul_3/ReadVariableOpCmodel_1/self_attn_model_1/private__attention_1/Mul_3/ReadVariableOp2�
Kmodel_1/spectral_normalization_14/conv2d_transpose_4/BiasAdd/ReadVariableOpKmodel_1/spectral_normalization_14/conv2d_transpose_4/BiasAdd/ReadVariableOp2�
Tmodel_1/spectral_normalization_14/conv2d_transpose_4/conv2d_transpose/ReadVariableOpTmodel_1/spectral_normalization_14/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2�
Kmodel_1/spectral_normalization_15/conv2d_transpose_5/BiasAdd/ReadVariableOpKmodel_1/spectral_normalization_15/conv2d_transpose_5/BiasAdd/ReadVariableOp2�
Tmodel_1/spectral_normalization_15/conv2d_transpose_5/conv2d_transpose/ReadVariableOpTmodel_1/spectral_normalization_15/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2�
Kmodel_1/spectral_normalization_16/conv2d_transpose_6/BiasAdd/ReadVariableOpKmodel_1/spectral_normalization_16/conv2d_transpose_6/BiasAdd/ReadVariableOp2�
Tmodel_1/spectral_normalization_16/conv2d_transpose_6/conv2d_transpose/ReadVariableOpTmodel_1/spectral_normalization_16/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2�
Kmodel_1/spectral_normalization_17/conv2d_transpose_7/BiasAdd/ReadVariableOpKmodel_1/spectral_normalization_17/conv2d_transpose_7/BiasAdd/ReadVariableOp2�
Tmodel_1/spectral_normalization_17/conv2d_transpose_7/conv2d_transpose/ReadVariableOpTmodel_1/spectral_normalization_17/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:P L
'
_output_shapes
:���������
!
_user_specified_name	input_4:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_148564

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
:__inference_spectral_normalization_14_layer_call_fn_149886

inputs"
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_149412x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:&"
 
_user_specified_name149880:&"
 
_user_specified_name149882
�X
�
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_150475

inputsB
(conv2d_11_conv2d_readvariableop_resource: 7
)conv2d_11_biasadd_readvariableop_resource:B
(conv2d_12_conv2d_readvariableop_resource: 7
)conv2d_12_biasadd_readvariableop_resource:B
(conv2d_13_conv2d_readvariableop_resource:  7
)conv2d_13_biasadd_readvariableop_resource: <
2private__attention_1_mul_3_readvariableop_resource: 
identity

identity_1�� conv2d_11/BiasAdd/ReadVariableOp�conv2d_11/Conv2D/ReadVariableOp� conv2d_12/BiasAdd/ReadVariableOp�conv2d_12/Conv2D/ReadVariableOp� conv2d_13/BiasAdd/ReadVariableOp�conv2d_13/Conv2D/ReadVariableOp�)private__attention_1/Mul_3/ReadVariableOp�
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_11/Conv2DConv2Dinputs'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_12/Conv2DConv2Dinputs'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
conv2d_13/Conv2DConv2Dinputs'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� r
private__attention_1/ShapeShapeconv2d_11/BiasAdd:output:0*
T0*
_output_shapes
::��r
(private__attention_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*private__attention_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*private__attention_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"private__attention_1/strided_sliceStridedSlice#private__attention_1/Shape:output:01private__attention_1/strided_slice/stack:output:03private__attention_1/strided_slice/stack_1:output:03private__attention_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
*private__attention_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,private__attention_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,private__attention_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$private__attention_1/strided_slice_1StridedSlice#private__attention_1/Shape:output:03private__attention_1/strided_slice_1/stack:output:05private__attention_1/strided_slice_1/stack_1:output:05private__attention_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
*private__attention_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,private__attention_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,private__attention_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$private__attention_1/strided_slice_2StridedSlice#private__attention_1/Shape:output:03private__attention_1/strided_slice_2/stack:output:05private__attention_1/strided_slice_2/stack_1:output:05private__attention_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
private__attention_1/mulMul-private__attention_1/strided_slice_1:output:0-private__attention_1/strided_slice_2:output:0*
T0*
_output_shapes
: o
$private__attention_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
����������
"private__attention_1/Reshape/shapePack+private__attention_1/strided_slice:output:0private__attention_1/mul:z:0-private__attention_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
private__attention_1/ReshapeReshapeconv2d_11/BiasAdd:output:0+private__attention_1/Reshape/shape:output:0*
T0*4
_output_shapes"
 :���������D����������
private__attention_1/mul_1Mul-private__attention_1/strided_slice_1:output:0-private__attention_1/strided_slice_2:output:0*
T0*
_output_shapes
: q
&private__attention_1/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
����������
$private__attention_1/Reshape_1/shapePack+private__attention_1/strided_slice:output:0private__attention_1/mul_1:z:0/private__attention_1/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:�
private__attention_1/Reshape_1Reshapeconv2d_12/BiasAdd:output:0-private__attention_1/Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :���������D���������x
#private__attention_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
private__attention_1/transpose	Transpose'private__attention_1/Reshape_1:output:0,private__attention_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D�
private__attention_1/mul_2Mul-private__attention_1/strided_slice_1:output:0-private__attention_1/strided_slice_2:output:0*
T0*
_output_shapes
: q
&private__attention_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
����������
$private__attention_1/Reshape_2/shapePack+private__attention_1/strided_slice:output:0private__attention_1/mul_2:z:0/private__attention_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:�
private__attention_1/Reshape_2Reshapeconv2d_13/BiasAdd:output:0-private__attention_1/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :���������D���������z
%private__attention_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
 private__attention_1/transpose_1	Transpose'private__attention_1/Reshape_2:output:0.private__attention_1/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������D�
private__attention_1/MatMulBatchMatMulV2%private__attention_1/Reshape:output:0"private__attention_1/transpose:y:0*
T0*+
_output_shapes
:���������DD�
private__attention_1/SoftmaxSoftmax$private__attention_1/MatMul:output:0*
T0*+
_output_shapes
:���������DDz
%private__attention_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
 private__attention_1/transpose_2	Transpose&private__attention_1/Softmax:softmax:0.private__attention_1/transpose_2/perm:output:0*
T0*+
_output_shapes
:���������DD�
private__attention_1/MatMul_1BatchMatMulV2$private__attention_1/transpose_1:y:0$private__attention_1/transpose_2:y:0*
T0*4
_output_shapes"
 :������������������Dz
%private__attention_1/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
 private__attention_1/transpose_3	Transpose&private__attention_1/MatMul_1:output:0.private__attention_1/transpose_3/perm:output:0*
T0*4
_output_shapes"
 :���������D���������q
&private__attention_1/Reshape_3/shape/3Const*
_output_shapes
: *
dtype0*
valueB :
����������
$private__attention_1/Reshape_3/shapePack+private__attention_1/strided_slice:output:0-private__attention_1/strided_slice_1:output:0-private__attention_1/strided_slice_2:output:0/private__attention_1/Reshape_3/shape/3:output:0*
N*
T0*
_output_shapes
:�
private__attention_1/Reshape_3Reshape$private__attention_1/transpose_3:y:0-private__attention_1/Reshape_3/shape:output:0*
T0*8
_output_shapes&
$:"�������������������
)private__attention_1/Mul_3/ReadVariableOpReadVariableOp2private__attention_1_mul_3_readvariableop_resource*
_output_shapes
: *
dtype0�
private__attention_1/Mul_3Mul'private__attention_1/Reshape_3:output:01private__attention_1/Mul_3/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"�������������������
private__attention_1/AddAddV2private__attention_1/Mul_3:z:0inputs*
T0*/
_output_shapes
:��������� s
IdentityIdentityprivate__attention_1/Add:z:0^NoOp*
T0*/
_output_shapes
:��������� {

Identity_1Identity&private__attention_1/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:���������DD�
NoOpNoOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp*^private__attention_1/Mul_3/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):��������� : : : : : : : 2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2V
)private__attention_1/Mul_3/ReadVariableOp)private__attention_1/Mul_3/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�H
�
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_149201

inputs:
reshape_readvariableop_resource:@�D
1spectral_normalize_matmul_readvariableop_resource:	�@
2conv2d_transpose_5_biasadd_readvariableop_resource:@
identity��Reshape/ReadVariableOp�)conv2d_transpose_5/BiasAdd/ReadVariableOp�2conv2d_transpose_5/conv2d_transpose/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOp
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*'
_output_shapes
:@�*
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   u
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0* 
_output_shapes
:
���
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes
:	�*
transpose_b(
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	�v
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
:	��
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes
:	��
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes
:	�x
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
:	��
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes
:	��
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	��
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes
:	��
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
!spectral_normalize/ReadVariableOpReadVariableOpreshape_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*'
_output_shapes
:@�y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   �   �
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*'
_output_shapes
:@��
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(\
conv2d_transpose_5/ShapeShapeinputs*
T0*
_output_shapes
::��p
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :	\
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*'
_output_shapes
:@�*
dtype0�
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:���������	@*
paddingSAME*
strides
�
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	@z
IdentityIdentity#conv2d_transpose_5/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������	@�
NoOpNoOp^Reshape/ReadVariableOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":����������: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_149961

inputsV
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource:	�A
2conv2d_transpose_4_biasadd_readvariableop_resource:	�
identity��)conv2d_transpose_4/BiasAdd/ReadVariableOp�2conv2d_transpose_4/conv2d_transpose/ReadVariableOp\
conv2d_transpose_4/ShapeShapeinputs*
T0*
_output_shapes
::��p
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*'
_output_shapes
:	�*
dtype0�
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������{
IdentityIdentity#conv2d_transpose_4/BiasAdd:output:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
(__inference_model_1_layer_call_fn_149687
input_4"
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�$
	unknown_5:@�
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11: @

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: 

unknown_18:$

unknown_19: 

unknown_20:$

unknown_21:  

unknown_22: 

unknown_23: $

unknown_24: 

unknown_25:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25*'
Tin 
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*=
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_149561w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_4:&"
 
_user_specified_name149631:&"
 
_user_specified_name149633:&"
 
_user_specified_name149635:&"
 
_user_specified_name149637:&"
 
_user_specified_name149639:&"
 
_user_specified_name149641:&"
 
_user_specified_name149643:&"
 
_user_specified_name149645:&	"
 
_user_specified_name149647:&
"
 
_user_specified_name149649:&"
 
_user_specified_name149651:&"
 
_user_specified_name149653:&"
 
_user_specified_name149655:&"
 
_user_specified_name149657:&"
 
_user_specified_name149659:&"
 
_user_specified_name149661:&"
 
_user_specified_name149663:&"
 
_user_specified_name149665:&"
 
_user_specified_name149667:&"
 
_user_specified_name149669:&"
 
_user_specified_name149671:&"
 
_user_specified_name149673:&"
 
_user_specified_name149675:&"
 
_user_specified_name149677:&"
 
_user_specified_name149679:&"
 
_user_specified_name149681:&"
 
_user_specified_name149683
�
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_148668

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
E__inference_conv2d_13_layer_call_and_return_conditional_losses_148840

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:��������� S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�Q
�
U__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_149377

inputs9
reshape_readvariableop_resource: C
1spectral_normalize_matmul_readvariableop_resource: @
2conv2d_transpose_7_biasadd_readvariableop_resource:
identity��Reshape/ReadVariableOp�)conv2d_transpose_7/BiasAdd/ReadVariableOp�2conv2d_transpose_7/conv2d_transpose/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
: *
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����    s
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes

: �
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes

:*
transpose_b(~
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes

:v
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

:�
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

:�
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
: *
dtype0�
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
: y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
: �
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(\
conv2d_transpose_7/ShapeShapeinputs*
T0*
_output_shapes
::��p
&conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_7/strided_sliceStridedSlice!conv2d_transpose_7/Shape:output:0/conv2d_transpose_7/strided_slice/stack:output:01conv2d_transpose_7/strided_slice/stack_1:output:01conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
conv2d_transpose_7/stackPack)conv2d_transpose_7/strided_slice:output:0#conv2d_transpose_7/stack/1:output:0#conv2d_transpose_7/stack/2:output:0#conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_7/strided_slice_1StridedSlice!conv2d_transpose_7/stack:output:01conv2d_transpose_7/strided_slice_1/stack:output:03conv2d_transpose_7/strided_slice_1/stack_1:output:03conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
: *
dtype0�
#conv2d_transpose_7/conv2d_transposeConv2DBackpropInput!conv2d_transpose_7/stack:output:0:conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
)conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_transpose_7/BiasAddBiasAdd,conv2d_transpose_7/conv2d_transpose:output:01conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������]
conv2d_transpose_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>_
conv2d_transpose_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?�
conv2d_transpose_7/MulMul#conv2d_transpose_7/BiasAdd:output:0!conv2d_transpose_7/Const:output:0*
T0*/
_output_shapes
:����������
conv2d_transpose_7/AddAddV2conv2d_transpose_7/Mul:z:0#conv2d_transpose_7/Const_1:output:0*
T0*/
_output_shapes
:���������o
*conv2d_transpose_7/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
(conv2d_transpose_7/clip_by_value/MinimumMinimumconv2d_transpose_7/Add:z:03conv2d_transpose_7/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:���������g
"conv2d_transpose_7/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
 conv2d_transpose_7/clip_by_valueMaximum,conv2d_transpose_7/clip_by_value/Minimum:z:0+conv2d_transpose_7/clip_by_value/y:output:0*
T0*/
_output_shapes
:���������{
IdentityIdentity$conv2d_transpose_7/clip_by_value:z:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp^Reshape/ReadVariableOp*^conv2d_transpose_7/BiasAdd/ReadVariableOp3^conv2d_transpose_7/conv2d_transpose/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:��������� : : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2V
)conv2d_transpose_7/BiasAdd/ReadVariableOp)conv2d_transpose_7/BiasAdd/ReadVariableOp2h
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�H
�
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_149127

inputs:
reshape_readvariableop_resource:	�C
1spectral_normalize_matmul_readvariableop_resource:A
2conv2d_transpose_4_biasadd_readvariableop_resource:	�
identity��Reshape/ReadVariableOp�)conv2d_transpose_4/BiasAdd/ReadVariableOp�2conv2d_transpose_4/conv2d_transpose/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOp
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*'
_output_shapes
:	�*
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   t
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	�$�
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes
:	�$*
transpose_b(
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	�$v
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
:	�$�
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

:�
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	�$�
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
!spectral_normalize/ReadVariableOpReadVariableOpreshape_readvariableop_resource*'
_output_shapes
:	�*
dtype0�
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*'
_output_shapes
:	�y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   	   �      �
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*'
_output_shapes
:	��
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(\
conv2d_transpose_4/ShapeShapeinputs*
T0*
_output_shapes
::��p
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*'
_output_shapes
:	�*
dtype0�
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������{
IdentityIdentity#conv2d_transpose_4/BiasAdd:output:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp^Reshape/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_149450

inputsV
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource:@�@
2conv2d_transpose_5_biasadd_readvariableop_resource:@
identity��)conv2d_transpose_5/BiasAdd/ReadVariableOp�2conv2d_transpose_5/conv2d_transpose/ReadVariableOp\
conv2d_transpose_5/ShapeShapeinputs*
T0*
_output_shapes
::��p
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :	\
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:���������	@*
paddingSAME*
strides
�
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	@z
IdentityIdentity#conv2d_transpose_5/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������	@�
NoOpNoOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
_
C__inference_re_lu_3_layer_call_and_return_conditional_losses_149148

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:����������c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�Y
�
C__inference_model_1_layer_call_and_return_conditional_losses_149386
input_4;
 spectral_normalization_14_149128:	�2
 spectral_normalization_14_149130:/
 spectral_normalization_14_149132:	�+
batch_normalization_3_149135:	�+
batch_normalization_3_149137:	�+
batch_normalization_3_149139:	�+
batch_normalization_3_149141:	�;
 spectral_normalization_15_149202:@�3
 spectral_normalization_15_149204:	�.
 spectral_normalization_15_149206:@*
batch_normalization_4_149209:@*
batch_normalization_4_149211:@*
batch_normalization_4_149213:@*
batch_normalization_4_149215:@:
 spectral_normalization_16_149276: @2
 spectral_normalization_16_149278:@.
 spectral_normalization_16_149280: *
batch_normalization_5_149287: *
batch_normalization_5_149289: *
batch_normalization_5_149291: *
batch_normalization_5_149293: 2
self_attn_model_1_149302: &
self_attn_model_1_149304:2
self_attn_model_1_149306: &
self_attn_model_1_149308:2
self_attn_model_1_149310:  &
self_attn_model_1_149312: "
self_attn_model_1_149314: :
 spectral_normalization_17_149378: 2
 spectral_normalization_17_149380: .
 spectral_normalization_17_149382:
identity��-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�)self_attn_model_1/StatefulPartitionedCall�1spectral_normalization_14/StatefulPartitionedCall�1spectral_normalization_15/StatefulPartitionedCall�1spectral_normalization_16/StatefulPartitionedCall�1spectral_normalization_17/StatefulPartitionedCall�
reshape_1/PartitionedCallPartitionedCallinput_4*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_149074�
1spectral_normalization_14/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0 spectral_normalization_14_149128 spectral_normalization_14_149130 spectral_normalization_14_149132*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_149127�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_14/StatefulPartitionedCall:output:0batch_normalization_3_149135batch_normalization_3_149137batch_normalization_3_149139batch_normalization_3_149141*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_148546�
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_149148�
1spectral_normalization_15/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0 spectral_normalization_15_149202 spectral_normalization_15_149204 spectral_normalization_15_149206*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_149201�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_15/StatefulPartitionedCall:output:0batch_normalization_4_149209batch_normalization_4_149211batch_normalization_4_149213batch_normalization_4_149215*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_148650�
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_re_lu_4_layer_call_and_return_conditional_losses_149222�
1spectral_normalization_16/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0 spectral_normalization_16_149276 spectral_normalization_16_149278 spectral_normalization_16_149280*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_149275�
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                �
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               �
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            �
(tf.__operators__.getitem_1/strided_sliceStridedSlice:spectral_normalization_16/StatefulPartitionedCall:output:07tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:��������� *

begin_mask*
end_mask�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall1tf.__operators__.getitem_1/strided_slice:output:0batch_normalization_5_149287batch_normalization_5_149289batch_normalization_5_149291batch_normalization_5_149293*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_148754�
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_re_lu_5_layer_call_and_return_conditional_losses_149300�
)self_attn_model_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0self_attn_model_1_149302self_attn_model_1_149304self_attn_model_1_149306self_attn_model_1_149308self_attn_model_1_149310self_attn_model_1_149312self_attn_model_1_149314*
Tin

2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:��������� :���������DD*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_148903�
1spectral_normalization_17/StatefulPartitionedCallStatefulPartitionedCall2self_attn_model_1/StatefulPartitionedCall:output:0 spectral_normalization_17_149378 spectral_normalization_17_149380 spectral_normalization_17_149382*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_149377�
IdentityIdentity:spectral_normalization_17/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall*^self_attn_model_1/StatefulPartitionedCall2^spectral_normalization_14/StatefulPartitionedCall2^spectral_normalization_15/StatefulPartitionedCall2^spectral_normalization_16/StatefulPartitionedCall2^spectral_normalization_17/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2V
)self_attn_model_1/StatefulPartitionedCall)self_attn_model_1/StatefulPartitionedCall2f
1spectral_normalization_14/StatefulPartitionedCall1spectral_normalization_14/StatefulPartitionedCall2f
1spectral_normalization_15/StatefulPartitionedCall1spectral_normalization_15/StatefulPartitionedCall2f
1spectral_normalization_16/StatefulPartitionedCall1spectral_normalization_16/StatefulPartitionedCall2f
1spectral_normalization_17/StatefulPartitionedCall1spectral_normalization_17/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_4:&"
 
_user_specified_name149128:&"
 
_user_specified_name149130:&"
 
_user_specified_name149132:&"
 
_user_specified_name149135:&"
 
_user_specified_name149137:&"
 
_user_specified_name149139:&"
 
_user_specified_name149141:&"
 
_user_specified_name149202:&	"
 
_user_specified_name149204:&
"
 
_user_specified_name149206:&"
 
_user_specified_name149209:&"
 
_user_specified_name149211:&"
 
_user_specified_name149213:&"
 
_user_specified_name149215:&"
 
_user_specified_name149276:&"
 
_user_specified_name149278:&"
 
_user_specified_name149280:&"
 
_user_specified_name149287:&"
 
_user_specified_name149289:&"
 
_user_specified_name149291:&"
 
_user_specified_name149293:&"
 
_user_specified_name149302:&"
 
_user_specified_name149304:&"
 
_user_specified_name149306:&"
 
_user_specified_name149308:&"
 
_user_specified_name149310:&"
 
_user_specified_name149312:&"
 
_user_specified_name149314:&"
 
_user_specified_name149378:&"
 
_user_specified_name149380:&"
 
_user_specified_name149382
�
_
C__inference_re_lu_5_layer_call_and_return_conditional_losses_150367

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:��������� b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
:__inference_spectral_normalization_17_layer_call_fn_150552

inputs!
unknown: 
	unknown_0: 
	unknown_1:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_149377w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:��������� : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:&"
 
_user_specified_name150544:&"
 
_user_specified_name150546:&"
 
_user_specified_name150548
�
�
2__inference_self_attn_model_1_layer_call_fn_150409

inputs!
unknown: 
	unknown_0:#
	unknown_1: 
	unknown_2:#
	unknown_3:  
	unknown_4: 
	unknown_5: 
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:��������� :���������DD*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_148927w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� u

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*+
_output_shapes
:���������DD<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):��������� : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:&"
 
_user_specified_name150391:&"
 
_user_specified_name150393:&"
 
_user_specified_name150395:&"
 
_user_specified_name150397:&"
 
_user_specified_name150399:&"
 
_user_specified_name150401:&"
 
_user_specified_name150403
�&
�
U__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_149554

inputsU
;conv2d_transpose_7_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_7_biasadd_readvariableop_resource:
identity��)conv2d_transpose_7/BiasAdd/ReadVariableOp�2conv2d_transpose_7/conv2d_transpose/ReadVariableOp\
conv2d_transpose_7/ShapeShapeinputs*
T0*
_output_shapes
::��p
&conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_7/strided_sliceStridedSlice!conv2d_transpose_7/Shape:output:0/conv2d_transpose_7/strided_slice/stack:output:01conv2d_transpose_7/strided_slice/stack_1:output:01conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
conv2d_transpose_7/stackPack)conv2d_transpose_7/strided_slice:output:0#conv2d_transpose_7/stack/1:output:0#conv2d_transpose_7/stack/2:output:0#conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_7/strided_slice_1StridedSlice!conv2d_transpose_7/stack:output:01conv2d_transpose_7/strided_slice_1/stack:output:03conv2d_transpose_7/strided_slice_1/stack_1:output:03conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
#conv2d_transpose_7/conv2d_transposeConv2DBackpropInput!conv2d_transpose_7/stack:output:0:conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
)conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_transpose_7/BiasAddBiasAdd,conv2d_transpose_7/conv2d_transpose:output:01conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������]
conv2d_transpose_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>_
conv2d_transpose_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?�
conv2d_transpose_7/MulMul#conv2d_transpose_7/BiasAdd:output:0!conv2d_transpose_7/Const:output:0*
T0*/
_output_shapes
:����������
conv2d_transpose_7/AddAddV2conv2d_transpose_7/Mul:z:0#conv2d_transpose_7/Const_1:output:0*
T0*/
_output_shapes
:���������o
*conv2d_transpose_7/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
(conv2d_transpose_7/clip_by_value/MinimumMinimumconv2d_transpose_7/Add:z:03conv2d_transpose_7/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:���������g
"conv2d_transpose_7/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
 conv2d_transpose_7/clip_by_valueMaximum,conv2d_transpose_7/clip_by_value/Minimum:z:0+conv2d_transpose_7/clip_by_value/y:output:0*
T0*/
_output_shapes
:���������{
IdentityIdentity$conv2d_transpose_7/clip_by_value:z:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp*^conv2d_transpose_7/BiasAdd/ReadVariableOp3^conv2d_transpose_7/conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 2V
)conv2d_transpose_7/BiasAdd/ReadVariableOp)conv2d_transpose_7/BiasAdd/ReadVariableOp2h
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2conv2d_transpose_7/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�!
�
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_150694

inputsC
(conv2d_transpose_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
shrink_axis_mask_
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
shrink_axis_mask_
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
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :�y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:	�*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
:__inference_spectral_normalization_16_layer_call_fn_150211

inputs!
unknown: @
	unknown_0:@
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
:��������� *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_149275w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������	@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������	@
 
_user_specified_nameinputs:&"
 
_user_specified_name150203:&"
 
_user_specified_name150205:&"
 
_user_specified_name150207
�

�
6__inference_batch_normalization_4_layer_call_fn_150154

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_148668�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:&"
 
_user_specified_name150144:&"
 
_user_specified_name150146:&"
 
_user_specified_name150148:&"
 
_user_specified_name150150
�
�
*__inference_conv2d_12_layer_call_fn_150869

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_148825w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:&"
 
_user_specified_name150863:&"
 
_user_specified_name150865
�!
�
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_150778

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
shrink_axis_mask_
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
shrink_axis_mask_
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
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� ]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_148546

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
:__inference_spectral_normalization_15_layer_call_fn_150053

inputs"
unknown:@�
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
:���������	@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_149450w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������	@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:&"
 
_user_specified_name150047:&"
 
_user_specified_name150049
�
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_150172

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
:__inference_spectral_normalization_16_layer_call_fn_150220

inputs!
unknown: @
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
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_149488w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������	@
 
_user_specified_nameinputs:&"
 
_user_specified_name150214:&"
 
_user_specified_name150216
�
D
(__inference_re_lu_4_layer_call_fn_150195

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_re_lu_4_layer_call_and_return_conditional_losses_149222h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������	@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	@:W S
/
_output_shapes
:���������	@
 
_user_specified_nameinputs
�H
�
U__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_150272

inputs9
reshape_readvariableop_resource: @C
1spectral_normalize_matmul_readvariableop_resource:@@
2conv2d_transpose_6_biasadd_readvariableop_resource: 
identity��Reshape/ReadVariableOp�)conv2d_transpose_6/BiasAdd/ReadVariableOp�2conv2d_transpose_6/conv2d_transpose/ReadVariableOp�#spectral_normalize/AssignVariableOp�%spectral_normalize/AssignVariableOp_1�(spectral_normalize/MatMul/ReadVariableOp�!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
: @*
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   s
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes

:`@�
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:@*
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

:`�
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
: @*
dtype0�
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
: @y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"          @   �
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
: @�
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(\
conv2d_transpose_6/ShapeShapeinputs*
T0*
_output_shapes
::��p
&conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_6/strided_sliceStridedSlice!conv2d_transpose_6/Shape:output:0/conv2d_transpose_6/strided_slice/stack:output:01conv2d_transpose_6/strided_slice/stack_1:output:01conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
conv2d_transpose_6/stackPack)conv2d_transpose_6/strided_slice:output:0#conv2d_transpose_6/stack/1:output:0#conv2d_transpose_6/stack/2:output:0#conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_6/strided_slice_1StridedSlice!conv2d_transpose_6/stack:output:01conv2d_transpose_6/strided_slice_1/stack:output:03conv2d_transpose_6/strided_slice_1/stack_1:output:03conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
: @*
dtype0�
#conv2d_transpose_6/conv2d_transposeConv2DBackpropInput!conv2d_transpose_6/stack:output:0:conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
)conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_transpose_6/BiasAddBiasAdd,conv2d_transpose_6/conv2d_transpose:output:01conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� z
IdentityIdentity#conv2d_transpose_6/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:��������� �
NoOpNoOp^Reshape/ReadVariableOp*^conv2d_transpose_6/BiasAdd/ReadVariableOp3^conv2d_transpose_6/conv2d_transpose/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������	@: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2V
)conv2d_transpose_6/BiasAdd/ReadVariableOp)conv2d_transpose_6/BiasAdd/ReadVariableOp2h
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:���������	@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
2__inference_self_attn_model_1_layer_call_fn_150388

inputs!
unknown: 
	unknown_0:#
	unknown_1: 
	unknown_2:#
	unknown_3:  
	unknown_4: 
	unknown_5: 
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:��������� :���������DD*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_148903w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� u

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*+
_output_shapes
:���������DD<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):��������� : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:&"
 
_user_specified_name150370:&"
 
_user_specified_name150372:&"
 
_user_specified_name150374:&"
 
_user_specified_name150376:&"
 
_user_specified_name150378:&"
 
_user_specified_name150380:&"
 
_user_specified_name150382
�	
�
:__inference_spectral_normalization_15_layer_call_fn_150044

inputs"
unknown:@�
	unknown_0:	�
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
:���������	@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_149201w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������	@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":����������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:&"
 
_user_specified_name150036:&"
 
_user_specified_name150038:&"
 
_user_specified_name150040
�
�
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_148927

inputs*
conv2d_11_148906: 
conv2d_11_148908:*
conv2d_12_148911: 
conv2d_12_148913:*
conv2d_13_148916:  
conv2d_13_148918: %
private__attention_1_148921: 
identity

identity_1��!conv2d_11/StatefulPartitionedCall�!conv2d_12/StatefulPartitionedCall�!conv2d_13/StatefulPartitionedCall�,private__attention_1/StatefulPartitionedCall�
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_11_148906conv2d_11_148908*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_148810�
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_12_148911conv2d_12_148913*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_148825�
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_13_148916conv2d_13_148918*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_148840�
,private__attention_1/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*conv2d_12/StatefulPartitionedCall:output:0*conv2d_13/StatefulPartitionedCall:output:0inputsprivate__attention_1_148921*
Tin	
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:��������� :���������DD*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_private__attention_1_layer_call_and_return_conditional_losses_148896�
IdentityIdentity5private__attention_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� �

Identity_1Identity5private__attention_1/StatefulPartitionedCall:output:1^NoOp*
T0*+
_output_shapes
:���������DD�
NoOpNoOp"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall-^private__attention_1/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):��������� : : : : : : : 2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2\
,private__attention_1/StatefulPartitionedCall,private__attention_1/StatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:&"
 
_user_specified_name148906:&"
 
_user_specified_name148908:&"
 
_user_specified_name148911:&"
 
_user_specified_name148913:&"
 
_user_specified_name148916:&"
 
_user_specified_name148918:&"
 
_user_specified_name148921
�/
�
P__inference_private__attention_1_layer_call_and_return_conditional_losses_150841
inputs_0
inputs_1
inputs_2
inputs_3'
mul_3_readvariableop_resource: 
identity

identity_1��Mul_3/ReadVariableOpK
ShapeShapeinputs_0*
T0*
_output_shapes
::��]
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
shrink_axis_mask_
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
shrink_axis_mask_
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
shrink_axis_mask_
mulMulstrided_slice_1:output:0strided_slice_2:output:0*
T0*
_output_shapes
: Z
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
���������~
Reshape/shapePackstrided_slice:output:0mul:z:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:s
ReshapeReshapeinputs_0Reshape/shape:output:0*
T0*4
_output_shapes"
 :���������D���������a
mul_1Mulstrided_slice_1:output:0strided_slice_2:output:0*
T0*
_output_shapes
: \
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
����������
Reshape_1/shapePackstrided_slice:output:0	mul_1:z:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:w
	Reshape_1Reshapeinputs_1Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :���������D���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
	transpose	TransposeReshape_1:output:0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������Da
mul_2Mulstrided_slice_1:output:0strided_slice_2:output:0*
T0*
_output_shapes
: \
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
����������
Reshape_2/shapePackstrided_slice:output:0	mul_2:z:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:w
	Reshape_2Reshapeinputs_2Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :���������D���������e
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	TransposeReshape_2:output:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������Dn
MatMulBatchMatMulV2Reshape:output:0transpose:y:0*
T0*+
_output_shapes
:���������DDY
SoftmaxSoftmaxMatMul:output:0*
T0*+
_output_shapes
:���������DDe
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_2	TransposeSoftmax:softmax:0transpose_2/perm:output:0*
T0*+
_output_shapes
:���������DDz
MatMul_1BatchMatMulV2transpose_1:y:0transpose_2:y:0*
T0*4
_output_shapes"
 :������������������De
transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_3	TransposeMatMul_1:output:0transpose_3/perm:output:0*
T0*4
_output_shapes"
 :���������D���������\
Reshape_3/shape/3Const*
_output_shapes
: *
dtype0*
valueB :
����������
Reshape_3/shapePackstrided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0Reshape_3/shape/3:output:0*
N*
T0*
_output_shapes
:�
	Reshape_3Reshapetranspose_3:y:0Reshape_3/shape:output:0*
T0*8
_output_shapes&
$:"������������������j
Mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes
: *
dtype0�
Mul_3MulReshape_3:output:0Mul_3/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������[
AddAddV2	Mul_3:z:0inputs_3*
T0*/
_output_shapes
:��������� ^
IdentityIdentityAdd:z:0^NoOp*
T0*/
_output_shapes
:��������� f

Identity_1IdentitySoftmax:softmax:0^NoOp*
T0*+
_output_shapes
:���������DD9
NoOpNoOp^Mul_3/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesp
n:���������:���������:��������� :��������� : 2,
Mul_3/ReadVariableOpMul_3/ReadVariableOp:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs_0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs_1:YU
/
_output_shapes
:��������� 
"
_user_specified_name
inputs_2:YU
/
_output_shapes
:��������� 
"
_user_specified_name
inputs_3:($
"
_user_specified_name
resource
�
�
3__inference_conv2d_transpose_6_layer_call_fn_150745

inputs!
unknown: @
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_148727�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:&"
 
_user_specified_name150739:&"
 
_user_specified_name150741
�&
�
U__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_150652

inputsU
;conv2d_transpose_7_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_7_biasadd_readvariableop_resource:
identity��)conv2d_transpose_7/BiasAdd/ReadVariableOp�2conv2d_transpose_7/conv2d_transpose/ReadVariableOp\
conv2d_transpose_7/ShapeShapeinputs*
T0*
_output_shapes
::��p
&conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_7/strided_sliceStridedSlice!conv2d_transpose_7/Shape:output:0/conv2d_transpose_7/strided_slice/stack:output:01conv2d_transpose_7/strided_slice/stack_1:output:01conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
conv2d_transpose_7/stackPack)conv2d_transpose_7/strided_slice:output:0#conv2d_transpose_7/stack/1:output:0#conv2d_transpose_7/stack/2:output:0#conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_7/strided_slice_1StridedSlice!conv2d_transpose_7/stack:output:01conv2d_transpose_7/strided_slice_1/stack:output:03conv2d_transpose_7/strided_slice_1/stack_1:output:03conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
#conv2d_transpose_7/conv2d_transposeConv2DBackpropInput!conv2d_transpose_7/stack:output:0:conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
)conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_transpose_7/BiasAddBiasAdd,conv2d_transpose_7/conv2d_transpose:output:01conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������]
conv2d_transpose_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>_
conv2d_transpose_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?�
conv2d_transpose_7/MulMul#conv2d_transpose_7/BiasAdd:output:0!conv2d_transpose_7/Const:output:0*
T0*/
_output_shapes
:����������
conv2d_transpose_7/AddAddV2conv2d_transpose_7/Mul:z:0#conv2d_transpose_7/Const_1:output:0*
T0*/
_output_shapes
:���������o
*conv2d_transpose_7/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
(conv2d_transpose_7/clip_by_value/MinimumMinimumconv2d_transpose_7/Add:z:03conv2d_transpose_7/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:���������g
"conv2d_transpose_7/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
 conv2d_transpose_7/clip_by_valueMaximum,conv2d_transpose_7/clip_by_value/Minimum:z:0+conv2d_transpose_7/clip_by_value/y:output:0*
T0*/
_output_shapes
:���������{
IdentityIdentity$conv2d_transpose_7/clip_by_value:z:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp*^conv2d_transpose_7/BiasAdd/ReadVariableOp3^conv2d_transpose_7/conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 2V
)conv2d_transpose_7/BiasAdd/ReadVariableOp)conv2d_transpose_7/BiasAdd/ReadVariableOp2h
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2conv2d_transpose_7/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
$__inference_signature_wrapper_149847
input_4"
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�$
	unknown_5:@�
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11: @

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: 

unknown_18:$

unknown_19: 

unknown_20:$

unknown_21:  

unknown_22: 

unknown_23: $

unknown_24: 

unknown_25:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25*'
Tin 
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*=
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_148486w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_4:&"
 
_user_specified_name149791:&"
 
_user_specified_name149793:&"
 
_user_specified_name149795:&"
 
_user_specified_name149797:&"
 
_user_specified_name149799:&"
 
_user_specified_name149801:&"
 
_user_specified_name149803:&"
 
_user_specified_name149805:&	"
 
_user_specified_name149807:&
"
 
_user_specified_name149809:&"
 
_user_specified_name149811:&"
 
_user_specified_name149813:&"
 
_user_specified_name149815:&"
 
_user_specified_name149817:&"
 
_user_specified_name149819:&"
 
_user_specified_name149821:&"
 
_user_specified_name149823:&"
 
_user_specified_name149825:&"
 
_user_specified_name149827:&"
 
_user_specified_name149829:&"
 
_user_specified_name149831:&"
 
_user_specified_name149833:&"
 
_user_specified_name149835:&"
 
_user_specified_name149837:&"
 
_user_specified_name149839:&"
 
_user_specified_name149841:&"
 
_user_specified_name149843
�
D
(__inference_re_lu_3_layer_call_fn_150028

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_149148i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_conv2d_11_layer_call_and_return_conditional_losses_148810

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
6__inference_batch_normalization_3_layer_call_fn_149987

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_148564�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:&"
 
_user_specified_name149977:&"
 
_user_specified_name149979:&"
 
_user_specified_name149981:&"
 
_user_specified_name149983
�
�
3__inference_conv2d_transpose_5_layer_call_fn_150703

inputs"
unknown:@�
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_148623�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:&"
 
_user_specified_name150697:&"
 
_user_specified_name150699
�!
�
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_150736

inputsC
(conv2d_transpose_readvariableop_resource:@�-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
shrink_axis_mask_
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
shrink_axis_mask_
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
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�(
�
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_150948

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
shrink_axis_mask_
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
shrink_axis_mask_
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
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?z
Mul_2MulBiasAdd:output:0Const:output:0*
T0*A
_output_shapes/
-:+���������������������������u
AddAddV2	Mul_2:z:0Const_1:output:0*
T0*A
_output_shapes/
-:+���������������������������\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value/MinimumMinimumAdd:z:0 clip_by_value/Minimum/y:output:0*
T0*A
_output_shapes/
-:+���������������������������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*A
_output_shapes/
-:+���������������������������z
IdentityIdentityclip_by_value:z:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
E__inference_conv2d_11_layer_call_and_return_conditional_losses_150860

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_150023

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
3__inference_conv2d_transpose_7_layer_call_fn_150907

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_149049�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:&"
 
_user_specified_name150901:&"
 
_user_specified_name150903
�
�
3__inference_conv2d_transpose_4_layer_call_fn_150661

inputs"
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_148519�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:&"
 
_user_specified_name150655:&"
 
_user_specified_name150657
�
D
(__inference_re_lu_5_layer_call_fn_150362

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_re_lu_5_layer_call_and_return_conditional_losses_149300h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
6__inference_batch_normalization_5_layer_call_fn_150321

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_148772�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:&"
 
_user_specified_name150311:&"
 
_user_specified_name150313:&"
 
_user_specified_name150315:&"
 
_user_specified_name150317
�
F
*__inference_reshape_1_layer_call_fn_149852

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_149074h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
U__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_150295

inputsU
;conv2d_transpose_6_conv2d_transpose_readvariableop_resource: @@
2conv2d_transpose_6_biasadd_readvariableop_resource: 
identity��)conv2d_transpose_6/BiasAdd/ReadVariableOp�2conv2d_transpose_6/conv2d_transpose/ReadVariableOp\
conv2d_transpose_6/ShapeShapeinputs*
T0*
_output_shapes
::��p
&conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_6/strided_sliceStridedSlice!conv2d_transpose_6/Shape:output:0/conv2d_transpose_6/strided_slice/stack:output:01conv2d_transpose_6/strided_slice/stack_1:output:01conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
conv2d_transpose_6/stackPack)conv2d_transpose_6/strided_slice:output:0#conv2d_transpose_6/stack/1:output:0#conv2d_transpose_6/stack/2:output:0#conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_6/strided_slice_1StridedSlice!conv2d_transpose_6/stack:output:01conv2d_transpose_6/strided_slice_1/stack:output:03conv2d_transpose_6/strided_slice_1/stack_1:output:03conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
#conv2d_transpose_6/conv2d_transposeConv2DBackpropInput!conv2d_transpose_6/stack:output:0:conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
)conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_transpose_6/BiasAddBiasAdd,conv2d_transpose_6/conv2d_transpose:output:01conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� z
IdentityIdentity#conv2d_transpose_6/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:��������� �
NoOpNoOp*^conv2d_transpose_6/BiasAdd/ReadVariableOp3^conv2d_transpose_6/conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	@: : 2V
)conv2d_transpose_6/BiasAdd/ReadVariableOp)conv2d_transpose_6/BiasAdd/ReadVariableOp2h
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2conv2d_transpose_6/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:���������	@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
(__inference_model_1_layer_call_fn_149628
input_4"
unknown:	�
	unknown_0:
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�$
	unknown_6:@�
	unknown_7:	�
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@$

unknown_13: @

unknown_14:@

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: 

unknown_19: $

unknown_20: 

unknown_21:$

unknown_22: 

unknown_23:$

unknown_24:  

unknown_25: 

unknown_26: $

unknown_27: 

unknown_28: 

unknown_29:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*3
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_149386w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_4:&"
 
_user_specified_name149564:&"
 
_user_specified_name149566:&"
 
_user_specified_name149568:&"
 
_user_specified_name149570:&"
 
_user_specified_name149572:&"
 
_user_specified_name149574:&"
 
_user_specified_name149576:&"
 
_user_specified_name149578:&	"
 
_user_specified_name149580:&
"
 
_user_specified_name149582:&"
 
_user_specified_name149584:&"
 
_user_specified_name149586:&"
 
_user_specified_name149588:&"
 
_user_specified_name149590:&"
 
_user_specified_name149592:&"
 
_user_specified_name149594:&"
 
_user_specified_name149596:&"
 
_user_specified_name149598:&"
 
_user_specified_name149600:&"
 
_user_specified_name149602:&"
 
_user_specified_name149604:&"
 
_user_specified_name149606:&"
 
_user_specified_name149608:&"
 
_user_specified_name149610:&"
 
_user_specified_name149612:&"
 
_user_specified_name149614:&"
 
_user_specified_name149616:&"
 
_user_specified_name149618:&"
 
_user_specified_name149620:&"
 
_user_specified_name149622:&"
 
_user_specified_name149624
�U
�
C__inference_model_1_layer_call_and_return_conditional_losses_149561
input_4;
 spectral_normalization_14_149413:	�/
 spectral_normalization_14_149415:	�+
batch_normalization_3_149418:	�+
batch_normalization_3_149420:	�+
batch_normalization_3_149422:	�+
batch_normalization_3_149424:	�;
 spectral_normalization_15_149451:@�.
 spectral_normalization_15_149453:@*
batch_normalization_4_149456:@*
batch_normalization_4_149458:@*
batch_normalization_4_149460:@*
batch_normalization_4_149462:@:
 spectral_normalization_16_149489: @.
 spectral_normalization_16_149491: *
batch_normalization_5_149498: *
batch_normalization_5_149500: *
batch_normalization_5_149502: *
batch_normalization_5_149504: 2
self_attn_model_1_149508: &
self_attn_model_1_149510:2
self_attn_model_1_149512: &
self_attn_model_1_149514:2
self_attn_model_1_149516:  &
self_attn_model_1_149518: "
self_attn_model_1_149520: :
 spectral_normalization_17_149555: .
 spectral_normalization_17_149557:
identity��-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�)self_attn_model_1/StatefulPartitionedCall�1spectral_normalization_14/StatefulPartitionedCall�1spectral_normalization_15/StatefulPartitionedCall�1spectral_normalization_16/StatefulPartitionedCall�1spectral_normalization_17/StatefulPartitionedCall�
reshape_1/PartitionedCallPartitionedCallinput_4*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_149074�
1spectral_normalization_14/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0 spectral_normalization_14_149413 spectral_normalization_14_149415*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_149412�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_14/StatefulPartitionedCall:output:0batch_normalization_3_149418batch_normalization_3_149420batch_normalization_3_149422batch_normalization_3_149424*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_148564�
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_149148�
1spectral_normalization_15/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0 spectral_normalization_15_149451 spectral_normalization_15_149453*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_149450�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_15/StatefulPartitionedCall:output:0batch_normalization_4_149456batch_normalization_4_149458batch_normalization_4_149460batch_normalization_4_149462*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_148668�
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_re_lu_4_layer_call_and_return_conditional_losses_149222�
1spectral_normalization_16/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0 spectral_normalization_16_149489 spectral_normalization_16_149491*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_149488�
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                �
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               �
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            �
(tf.__operators__.getitem_1/strided_sliceStridedSlice:spectral_normalization_16/StatefulPartitionedCall:output:07tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:��������� *

begin_mask*
end_mask�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall1tf.__operators__.getitem_1/strided_slice:output:0batch_normalization_5_149498batch_normalization_5_149500batch_normalization_5_149502batch_normalization_5_149504*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_148772�
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_re_lu_5_layer_call_and_return_conditional_losses_149300�
)self_attn_model_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0self_attn_model_1_149508self_attn_model_1_149510self_attn_model_1_149512self_attn_model_1_149514self_attn_model_1_149516self_attn_model_1_149518self_attn_model_1_149520*
Tin

2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:��������� :���������DD*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_148927�
1spectral_normalization_17/StatefulPartitionedCallStatefulPartitionedCall2self_attn_model_1/StatefulPartitionedCall:output:0 spectral_normalization_17_149555 spectral_normalization_17_149557*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_149554�
IdentityIdentity:spectral_normalization_17/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall*^self_attn_model_1/StatefulPartitionedCall2^spectral_normalization_14/StatefulPartitionedCall2^spectral_normalization_15/StatefulPartitionedCall2^spectral_normalization_16/StatefulPartitionedCall2^spectral_normalization_17/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2V
)self_attn_model_1/StatefulPartitionedCall)self_attn_model_1/StatefulPartitionedCall2f
1spectral_normalization_14/StatefulPartitionedCall1spectral_normalization_14/StatefulPartitionedCall2f
1spectral_normalization_15/StatefulPartitionedCall1spectral_normalization_15/StatefulPartitionedCall2f
1spectral_normalization_16/StatefulPartitionedCall1spectral_normalization_16/StatefulPartitionedCall2f
1spectral_normalization_17/StatefulPartitionedCall1spectral_normalization_17/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_4:&"
 
_user_specified_name149413:&"
 
_user_specified_name149415:&"
 
_user_specified_name149418:&"
 
_user_specified_name149420:&"
 
_user_specified_name149422:&"
 
_user_specified_name149424:&"
 
_user_specified_name149451:&"
 
_user_specified_name149453:&	"
 
_user_specified_name149456:&
"
 
_user_specified_name149458:&"
 
_user_specified_name149460:&"
 
_user_specified_name149462:&"
 
_user_specified_name149489:&"
 
_user_specified_name149491:&"
 
_user_specified_name149498:&"
 
_user_specified_name149500:&"
 
_user_specified_name149502:&"
 
_user_specified_name149504:&"
 
_user_specified_name149508:&"
 
_user_specified_name149510:&"
 
_user_specified_name149512:&"
 
_user_specified_name149514:&"
 
_user_specified_name149516:&"
 
_user_specified_name149518:&"
 
_user_specified_name149520:&"
 
_user_specified_name149555:&"
 
_user_specified_name149557
�
a
E__inference_reshape_1_layer_call_and_return_conditional_losses_149074

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
5__inference_private__attention_1_layer_call_fn_150790
inputs_0
inputs_1
inputs_2
inputs_3
unknown: 
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3unknown*
Tin	
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:��������� :���������DD*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_private__attention_1_layer_call_and_return_conditional_losses_148896w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� u

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*+
_output_shapes
:���������DD<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesp
n:���������:���������:��������� :��������� : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������
"
_user_specified_name
inputs_0:YU
/
_output_shapes
:���������
"
_user_specified_name
inputs_1:YU
/
_output_shapes
:��������� 
"
_user_specified_name
inputs_2:YU
/
_output_shapes
:��������� 
"
_user_specified_name
inputs_3:&"
 
_user_specified_name150784
�/
�
P__inference_private__attention_1_layer_call_and_return_conditional_losses_148896

inputs
inputs_1
inputs_2
inputs_3'
mul_3_readvariableop_resource: 
identity

identity_1��Mul_3/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
shrink_axis_mask_
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
shrink_axis_mask_
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
shrink_axis_mask_
mulMulstrided_slice_1:output:0strided_slice_2:output:0*
T0*
_output_shapes
: Z
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
���������~
Reshape/shapePackstrided_slice:output:0mul:z:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:q
ReshapeReshapeinputsReshape/shape:output:0*
T0*4
_output_shapes"
 :���������D���������a
mul_1Mulstrided_slice_1:output:0strided_slice_2:output:0*
T0*
_output_shapes
: \
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
����������
Reshape_1/shapePackstrided_slice:output:0	mul_1:z:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:w
	Reshape_1Reshapeinputs_1Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :���������D���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
	transpose	TransposeReshape_1:output:0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������Da
mul_2Mulstrided_slice_1:output:0strided_slice_2:output:0*
T0*
_output_shapes
: \
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
����������
Reshape_2/shapePackstrided_slice:output:0	mul_2:z:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:w
	Reshape_2Reshapeinputs_2Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :���������D���������e
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	TransposeReshape_2:output:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������Dn
MatMulBatchMatMulV2Reshape:output:0transpose:y:0*
T0*+
_output_shapes
:���������DDY
SoftmaxSoftmaxMatMul:output:0*
T0*+
_output_shapes
:���������DDe
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_2	TransposeSoftmax:softmax:0transpose_2/perm:output:0*
T0*+
_output_shapes
:���������DDz
MatMul_1BatchMatMulV2transpose_1:y:0transpose_2:y:0*
T0*4
_output_shapes"
 :������������������De
transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_3	TransposeMatMul_1:output:0transpose_3/perm:output:0*
T0*4
_output_shapes"
 :���������D���������\
Reshape_3/shape/3Const*
_output_shapes
: *
dtype0*
valueB :
����������
Reshape_3/shapePackstrided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0Reshape_3/shape/3:output:0*
N*
T0*
_output_shapes
:�
	Reshape_3Reshapetranspose_3:y:0Reshape_3/shape:output:0*
T0*8
_output_shapes&
$:"������������������j
Mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes
: *
dtype0�
Mul_3MulReshape_3:output:0Mul_3/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������[
AddAddV2	Mul_3:z:0inputs_3*
T0*/
_output_shapes
:��������� ^
IdentityIdentityAdd:z:0^NoOp*
T0*/
_output_shapes
:��������� f

Identity_1IdentitySoftmax:softmax:0^NoOp*
T0*+
_output_shapes
:���������DD9
NoOpNoOp^Mul_3/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesp
n:���������:���������:��������� :��������� : 2,
Mul_3/ReadVariableOpMul_3/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:���������
 
_user_specified_nameinputs:WS
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:WS
/
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource
�

�
6__inference_batch_normalization_3_layer_call_fn_149974

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_148546�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:&"
 
_user_specified_name149964:&"
 
_user_specified_name149966:&"
 
_user_specified_name149968:&"
 
_user_specified_name149970
�(
�
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_149049

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::��]
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
shrink_axis_mask_
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
shrink_axis_mask_
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
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��L>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?z
Mul_2MulBiasAdd:output:0Const:output:0*
T0*A
_output_shapes/
-:+���������������������������u
AddAddV2	Mul_2:z:0Const_1:output:0*
T0*A
_output_shapes/
-:+���������������������������\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_value/MinimumMinimumAdd:z:0 clip_by_value/Minimum/y:output:0*
T0*A
_output_shapes/
-:+���������������������������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*A
_output_shapes/
-:+���������������������������z
IdentityIdentityclip_by_value:z:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������]
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_150005

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
_
C__inference_re_lu_3_layer_call_and_return_conditional_losses_150033

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:����������c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
U__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_149488

inputsU
;conv2d_transpose_6_conv2d_transpose_readvariableop_resource: @@
2conv2d_transpose_6_biasadd_readvariableop_resource: 
identity��)conv2d_transpose_6/BiasAdd/ReadVariableOp�2conv2d_transpose_6/conv2d_transpose/ReadVariableOp\
conv2d_transpose_6/ShapeShapeinputs*
T0*
_output_shapes
::��p
&conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_6/strided_sliceStridedSlice!conv2d_transpose_6/Shape:output:0/conv2d_transpose_6/strided_slice/stack:output:01conv2d_transpose_6/strided_slice/stack_1:output:01conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
conv2d_transpose_6/stackPack)conv2d_transpose_6/strided_slice:output:0#conv2d_transpose_6/stack/1:output:0#conv2d_transpose_6/stack/2:output:0#conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_6/strided_slice_1StridedSlice!conv2d_transpose_6/stack:output:01conv2d_transpose_6/strided_slice_1/stack:output:03conv2d_transpose_6/strided_slice_1/stack_1:output:03conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
#conv2d_transpose_6/conv2d_transposeConv2DBackpropInput!conv2d_transpose_6/stack:output:0:conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
)conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_transpose_6/BiasAddBiasAdd,conv2d_transpose_6/conv2d_transpose:output:01conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� z
IdentityIdentity#conv2d_transpose_6/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:��������� �
NoOpNoOp*^conv2d_transpose_6/BiasAdd/ReadVariableOp3^conv2d_transpose_6/conv2d_transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	@: : 2V
)conv2d_transpose_6/BiasAdd/ReadVariableOp)conv2d_transpose_6/BiasAdd/ReadVariableOp2h
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2conv2d_transpose_6/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:���������	@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_40
serving_default_input_4:0���������U
spectral_normalization_178
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
	#layer
$w
%w_shape
&sn_u
&u"
_tf_keras_layer
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses
-axis
	.gamma
/beta
0moving_mean
1moving_variance"
_tf_keras_layer
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses"
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
Jbeta
Kmoving_mean
Lmoving_variance"
_tf_keras_layer
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
	Ylayer
Zw
[w_shape
\sn_u
\u"
_tf_keras_layer
(
]	keras_api"
_tf_keras_layer
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses
daxis
	egamma
fbeta
gmoving_mean
hmoving_variance"
_tf_keras_layer
�
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_layer
�
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses
uattn
v
query_conv
wkey_conv
x
value_conv"
_tf_keras_model
�
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses
	layer
�w
�w_shape
	�sn_u
�u"
_tf_keras_layer
�
$0
�1
&2
.3
/4
05
16
?7
�8
A9
I10
J11
K12
L13
Z14
�15
\16
e17
f18
g19
h20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30"
trackable_list_wrapper
�
$0
�1
.2
/3
?4
�5
I6
J7
Z8
�9
e10
f11
�12
�13
�14
�15
�16
�17
�18
�19
�20"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
(__inference_model_1_layer_call_fn_149628
(__inference_model_1_layer_call_fn_149687�
���
FullArgSpec)
args!�
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
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
C__inference_model_1_layer_call_and_return_conditional_losses_149386
C__inference_model_1_layer_call_and_return_conditional_losses_149561�
���
FullArgSpec)
args!�
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
annotations� *
 z�trace_0z�trace_1
�B�
!__inference__wrapped_model_148486input_4"�
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
-
�serving_default"
signature_map
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
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_reshape_1_layer_call_fn_149852�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
E__inference_reshape_1_layer_call_and_return_conditional_losses_149866�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
6
$0
�1
&2"
trackable_list_wrapper
/
$0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
:__inference_spectral_normalization_14_layer_call_fn_149877
:__inference_spectral_normalization_14_layer_call_fn_149886�
���
FullArgSpec!
args�
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
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_149938
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_149961�
���
FullArgSpec!
args�
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
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

$kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
;:9	�2 spectral_normalization_14/kernel
 "
trackable_list_wrapper
.:,2spectral_normalization_14/sn_u
<
.0
/1
02
13"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_3_layer_call_fn_149974
6__inference_batch_normalization_3_layer_call_fn_149987�
���
FullArgSpec)
args!�
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
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_150005
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_150023�
���
FullArgSpec)
args!�
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
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(�2batch_normalization_3/gamma
):'�2batch_normalization_3/beta
2:0� (2!batch_normalization_3/moving_mean
6:4� (2%batch_normalization_3/moving_variance
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
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_re_lu_3_layer_call_fn_150028�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
C__inference_re_lu_3_layer_call_and_return_conditional_losses_150033�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
6
?0
�1
A2"
trackable_list_wrapper
/
?0
�1"
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
:__inference_spectral_normalization_15_layer_call_fn_150044
:__inference_spectral_normalization_15_layer_call_fn_150053�
���
FullArgSpec!
args�
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
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_150105
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_150128�
���
FullArgSpec!
args�
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
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

?kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
;:9@�2 spectral_normalization_15/kernel
 "
trackable_list_wrapper
/:-	�2spectral_normalization_15/sn_u
<
I0
J1
K2
L3"
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
�
�trace_0
�trace_12�
6__inference_batch_normalization_4_layer_call_fn_150141
6__inference_batch_normalization_4_layer_call_fn_150154�
���
FullArgSpec)
args!�
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
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_150172
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_150190�
���
FullArgSpec)
args!�
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
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
):'@2batch_normalization_4/gamma
(:&@2batch_normalization_4/beta
1:/@ (2!batch_normalization_4/moving_mean
5:3@ (2%batch_normalization_4/moving_variance
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
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_re_lu_4_layer_call_fn_150195�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
C__inference_re_lu_4_layer_call_and_return_conditional_losses_150200�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
6
Z0
�1
\2"
trackable_list_wrapper
/
Z0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
:__inference_spectral_normalization_16_layer_call_fn_150211
:__inference_spectral_normalization_16_layer_call_fn_150220�
���
FullArgSpec!
args�
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
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
U__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_150272
U__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_150295�
���
FullArgSpec!
args�
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
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Zkernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
::8 @2 spectral_normalization_16/kernel
 "
trackable_list_wrapper
.:,@2spectral_normalization_16/sn_u
"
_generic_user_object
<
e0
f1
g2
h3"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_5_layer_call_fn_150308
6__inference_batch_normalization_5_layer_call_fn_150321�
���
FullArgSpec)
args!�
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
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_150339
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_150357�
���
FullArgSpec)
args!�
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
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
):' 2batch_normalization_5/gamma
(:& 2batch_normalization_5/beta
1:/  (2!batch_normalization_5/moving_mean
5:3  (2%batch_normalization_5/moving_variance
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
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_re_lu_5_layer_call_fn_150362�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
C__inference_re_lu_5_layer_call_and_return_conditional_losses_150367�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
X
�0
�1
�2
�3
�4
�5
�6"
trackable_list_wrapper
X
�0
�1
�2
�3
�4
�5
�6"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
2__inference_self_attn_model_1_layer_call_fn_150388
2__inference_self_attn_model_1_layer_call_fn_150409�
���
FullArgSpec!
args�
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
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_150475
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_150541�
���
FullArgSpec!
args�
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
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�private__attention_1_gamma

�gamma"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
8
�0
�1
�2"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
:__inference_spectral_normalization_17_layer_call_fn_150552
:__inference_spectral_normalization_17_layer_call_fn_150561�
���
FullArgSpec!
args�
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
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
U__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_150621
U__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_150652�
���
FullArgSpec!
args�
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
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
::8 2 spectral_normalization_17/kernel
 "
trackable_list_wrapper
.:, 2spectral_normalization_17/sn_u
-:+�2spectral_normalization_14/bias
,:*@2spectral_normalization_15/bias
,:* 2spectral_normalization_16/bias
K:I 2Aself_attn_model_1/private__attention_1/private__attention_1_gamma
<:: 2"self_attn_model_1/conv2d_11/kernel
.:,2 self_attn_model_1/conv2d_11/bias
<:: 2"self_attn_model_1/conv2d_12/kernel
.:,2 self_attn_model_1/conv2d_12/bias
<::  2"self_attn_model_1/conv2d_13/kernel
.:, 2 self_attn_model_1/conv2d_13/bias
,:*2spectral_normalization_17/bias
g
&0
01
12
A3
K4
L5
\6
g7
h8
�9"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_model_1_layer_call_fn_149628input_4"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
(__inference_model_1_layer_call_fn_149687input_4"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
C__inference_model_1_layer_call_and_return_conditional_losses_149386input_4"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
C__inference_model_1_layer_call_and_return_conditional_losses_149561input_4"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
$__inference_signature_wrapper_149847input_4"�
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
*__inference_reshape_1_layer_call_fn_149852inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
E__inference_reshape_1_layer_call_and_return_conditional_losses_149866inputs"�
���
FullArgSpec
args�

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
annotations� *
 
'
&0"
trackable_list_wrapper
'
#0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
:__inference_spectral_normalization_14_layer_call_fn_149877inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
�B�
:__inference_spectral_normalization_14_layer_call_fn_149886inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
�B�
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_149938inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
�B�
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_149961inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
/
$0
�1"
trackable_list_wrapper
/
$0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_conv2d_transpose_4_layer_call_fn_150661�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_150694�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
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
annotations� *
 0
.
00
11"
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
6__inference_batch_normalization_3_layer_call_fn_149974inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
6__inference_batch_normalization_3_layer_call_fn_149987inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_150005inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_150023inputs"�
���
FullArgSpec)
args!�
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
annotations� *
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
(__inference_re_lu_3_layer_call_fn_150028inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
C__inference_re_lu_3_layer_call_and_return_conditional_losses_150033inputs"�
���
FullArgSpec
args�

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
annotations� *
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
:__inference_spectral_normalization_15_layer_call_fn_150044inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
�B�
:__inference_spectral_normalization_15_layer_call_fn_150053inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
�B�
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_150105inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
�B�
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_150128inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
/
?0
�1"
trackable_list_wrapper
/
?0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_conv2d_transpose_5_layer_call_fn_150703�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_150736�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
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
annotations� *
 0
.
K0
L1"
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
6__inference_batch_normalization_4_layer_call_fn_150141inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
6__inference_batch_normalization_4_layer_call_fn_150154inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_150172inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_150190inputs"�
���
FullArgSpec)
args!�
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
annotations� *
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
(__inference_re_lu_4_layer_call_fn_150195inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
C__inference_re_lu_4_layer_call_and_return_conditional_losses_150200inputs"�
���
FullArgSpec
args�

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
annotations� *
 
'
\0"
trackable_list_wrapper
'
Y0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
:__inference_spectral_normalization_16_layer_call_fn_150211inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
�B�
:__inference_spectral_normalization_16_layer_call_fn_150220inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
�B�
U__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_150272inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
�B�
U__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_150295inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
/
Z0
�1"
trackable_list_wrapper
/
Z0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_conv2d_transpose_6_layer_call_fn_150745�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_150778�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
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
annotations� *
 0
.
g0
h1"
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
6__inference_batch_normalization_5_layer_call_fn_150308inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
6__inference_batch_normalization_5_layer_call_fn_150321inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_150339inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_150357inputs"�
���
FullArgSpec)
args!�
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
annotations� *
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
(__inference_re_lu_5_layer_call_fn_150362inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
C__inference_re_lu_5_layer_call_and_return_conditional_losses_150367inputs"�
���
FullArgSpec
args�

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
annotations� *
 
 "
trackable_list_wrapper
<
u0
v1
w2
x3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
2__inference_self_attn_model_1_layer_call_fn_150388inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
�B�
2__inference_self_attn_model_1_layer_call_fn_150409inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
�B�
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_150475inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
�B�
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_150541inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
5__inference_private__attention_1_layer_call_fn_150790�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
P__inference_private__attention_1_layer_call_and_return_conditional_losses_150841�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_11_layer_call_fn_150850�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_11_layer_call_and_return_conditional_losses_150860�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
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
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_12_layer_call_fn_150869�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_12_layer_call_and_return_conditional_losses_150879�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
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
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_13_layer_call_fn_150888�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_13_layer_call_and_return_conditional_losses_150898�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
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
annotations� *
 0
(
�0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
:__inference_spectral_normalization_17_layer_call_fn_150552inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
�B�
:__inference_spectral_normalization_17_layer_call_fn_150561inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
�B�
U__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_150621inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
�B�
U__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_150652inputs"�
���
FullArgSpec!
args�
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
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_conv2d_transpose_7_layer_call_fn_150907�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_150948�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
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
annotations� *
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
3__inference_conv2d_transpose_4_layer_call_fn_150661inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_150694inputs"�
���
FullArgSpec
args�

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
annotations� *
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
3__inference_conv2d_transpose_5_layer_call_fn_150703inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_150736inputs"�
���
FullArgSpec
args�

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
annotations� *
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
3__inference_conv2d_transpose_6_layer_call_fn_150745inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_150778inputs"�
���
FullArgSpec
args�

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
annotations� *
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
5__inference_private__attention_1_layer_call_fn_150790inputs_0inputs_1inputs_2inputs_3"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
P__inference_private__attention_1_layer_call_and_return_conditional_losses_150841inputs_0inputs_1inputs_2inputs_3"�
���
FullArgSpec
args�

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
annotations� *
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
*__inference_conv2d_11_layer_call_fn_150850inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
E__inference_conv2d_11_layer_call_and_return_conditional_losses_150860inputs"�
���
FullArgSpec
args�

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
annotations� *
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
*__inference_conv2d_12_layer_call_fn_150869inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
E__inference_conv2d_12_layer_call_and_return_conditional_losses_150879inputs"�
���
FullArgSpec
args�

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
annotations� *
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
*__inference_conv2d_13_layer_call_fn_150888inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
E__inference_conv2d_13_layer_call_and_return_conditional_losses_150898inputs"�
���
FullArgSpec
args�

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
annotations� *
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
3__inference_conv2d_transpose_7_layer_call_fn_150907inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_150948inputs"�
���
FullArgSpec
args�

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
annotations� *
 �
!__inference__wrapped_model_148486�'$�./01?�IJKLZ�efgh���������0�-
&�#
!�
input_4���������
� "]�Z
X
spectral_normalization_17;�8
spectral_normalization_17����������
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_150005�./01R�O
H�E
;�8
inputs,����������������������������
p

 
� "G�D
=�:
tensor_0,����������������������������
� �
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_150023�./01R�O
H�E
;�8
inputs,����������������������������
p 

 
� "G�D
=�:
tensor_0,����������������������������
� �
6__inference_batch_normalization_3_layer_call_fn_149974�./01R�O
H�E
;�8
inputs,����������������������������
p

 
� "<�9
unknown,�����������������������������
6__inference_batch_normalization_3_layer_call_fn_149987�./01R�O
H�E
;�8
inputs,����������������������������
p 

 
� "<�9
unknown,�����������������������������
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_150172�IJKLQ�N
G�D
:�7
inputs+���������������������������@
p

 
� "F�C
<�9
tensor_0+���������������������������@
� �
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_150190�IJKLQ�N
G�D
:�7
inputs+���������������������������@
p 

 
� "F�C
<�9
tensor_0+���������������������������@
� �
6__inference_batch_normalization_4_layer_call_fn_150141�IJKLQ�N
G�D
:�7
inputs+���������������������������@
p

 
� ";�8
unknown+���������������������������@�
6__inference_batch_normalization_4_layer_call_fn_150154�IJKLQ�N
G�D
:�7
inputs+���������������������������@
p 

 
� ";�8
unknown+���������������������������@�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_150339�efghQ�N
G�D
:�7
inputs+��������������������������� 
p

 
� "F�C
<�9
tensor_0+��������������������������� 
� �
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_150357�efghQ�N
G�D
:�7
inputs+��������������������������� 
p 

 
� "F�C
<�9
tensor_0+��������������������������� 
� �
6__inference_batch_normalization_5_layer_call_fn_150308�efghQ�N
G�D
:�7
inputs+��������������������������� 
p

 
� ";�8
unknown+��������������������������� �
6__inference_batch_normalization_5_layer_call_fn_150321�efghQ�N
G�D
:�7
inputs+��������������������������� 
p 

 
� ";�8
unknown+��������������������������� �
E__inference_conv2d_11_layer_call_and_return_conditional_losses_150860u��7�4
-�*
(�%
inputs��������� 
� "4�1
*�'
tensor_0���������
� �
*__inference_conv2d_11_layer_call_fn_150850j��7�4
-�*
(�%
inputs��������� 
� ")�&
unknown����������
E__inference_conv2d_12_layer_call_and_return_conditional_losses_150879u��7�4
-�*
(�%
inputs��������� 
� "4�1
*�'
tensor_0���������
� �
*__inference_conv2d_12_layer_call_fn_150869j��7�4
-�*
(�%
inputs��������� 
� ")�&
unknown����������
E__inference_conv2d_13_layer_call_and_return_conditional_losses_150898u��7�4
-�*
(�%
inputs��������� 
� "4�1
*�'
tensor_0��������� 
� �
*__inference_conv2d_13_layer_call_fn_150888j��7�4
-�*
(�%
inputs��������� 
� ")�&
unknown��������� �
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_150694�$�I�F
?�<
:�7
inputs+���������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
3__inference_conv2d_transpose_4_layer_call_fn_150661�$�I�F
?�<
:�7
inputs+���������������������������
� "<�9
unknown,�����������������������������
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_150736�?�J�G
@�=
;�8
inputs,����������������������������
� "F�C
<�9
tensor_0+���������������������������@
� �
3__inference_conv2d_transpose_5_layer_call_fn_150703�?�J�G
@�=
;�8
inputs,����������������������������
� ";�8
unknown+���������������������������@�
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_150778�Z�I�F
?�<
:�7
inputs+���������������������������@
� "F�C
<�9
tensor_0+��������������������������� 
� �
3__inference_conv2d_transpose_6_layer_call_fn_150745�Z�I�F
?�<
:�7
inputs+���������������������������@
� ";�8
unknown+��������������������������� �
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_150948���I�F
?�<
:�7
inputs+��������������������������� 
� "F�C
<�9
tensor_0+���������������������������
� �
3__inference_conv2d_transpose_7_layer_call_fn_150907���I�F
?�<
:�7
inputs+��������������������������� 
� ";�8
unknown+����������������������������
C__inference_model_1_layer_call_and_return_conditional_losses_149386�,$&�./01?A�IJKLZ\�efgh����������8�5
.�+
!�
input_4���������
p

 
� "4�1
*�'
tensor_0���������
� �
C__inference_model_1_layer_call_and_return_conditional_losses_149561�'$�./01?�IJKLZ�efgh���������8�5
.�+
!�
input_4���������
p 

 
� "4�1
*�'
tensor_0���������
� �
(__inference_model_1_layer_call_fn_149628�,$&�./01?A�IJKLZ\�efgh����������8�5
.�+
!�
input_4���������
p

 
� ")�&
unknown����������
(__inference_model_1_layer_call_fn_149687�'$�./01?�IJKLZ�efgh���������8�5
.�+
!�
input_4���������
p 

 
� ")�&
unknown����������
P__inference_private__attention_1_layer_call_and_return_conditional_losses_150841�����
���
���
*�'
inputs_0���������
*�'
inputs_1���������
*�'
inputs_2��������� 
*�'
inputs_3��������� 
� "e�b
[�X
,�)

tensor_0_0��������� 
(�%

tensor_0_1���������DD
� �
5__inference_private__attention_1_layer_call_fn_150790�����
���
���
*�'
inputs_0���������
*�'
inputs_1���������
*�'
inputs_2��������� 
*�'
inputs_3��������� 
� "W�T
*�'
tensor_0��������� 
&�#
tensor_1���������DD�
C__inference_re_lu_3_layer_call_and_return_conditional_losses_150033q8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
(__inference_re_lu_3_layer_call_fn_150028f8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
C__inference_re_lu_4_layer_call_and_return_conditional_losses_150200o7�4
-�*
(�%
inputs���������	@
� "4�1
*�'
tensor_0���������	@
� �
(__inference_re_lu_4_layer_call_fn_150195d7�4
-�*
(�%
inputs���������	@
� ")�&
unknown���������	@�
C__inference_re_lu_5_layer_call_and_return_conditional_losses_150367o7�4
-�*
(�%
inputs��������� 
� "4�1
*�'
tensor_0��������� 
� �
(__inference_re_lu_5_layer_call_fn_150362d7�4
-�*
(�%
inputs��������� 
� ")�&
unknown��������� �
E__inference_reshape_1_layer_call_and_return_conditional_losses_149866g/�,
%�"
 �
inputs���������
� "4�1
*�'
tensor_0���������
� �
*__inference_reshape_1_layer_call_fn_149852\/�,
%�"
 �
inputs���������
� ")�&
unknown����������
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_150475��������;�8
1�.
(�%
inputs��������� 
p
� "e�b
[�X
,�)

tensor_0_0��������� 
(�%

tensor_0_1���������DD
� �
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_150541��������;�8
1�.
(�%
inputs��������� 
p 
� "e�b
[�X
,�)

tensor_0_0��������� 
(�%

tensor_0_1���������DD
� �
2__inference_self_attn_model_1_layer_call_fn_150388��������;�8
1�.
(�%
inputs��������� 
p
� "W�T
*�'
tensor_0��������� 
&�#
tensor_1���������DD�
2__inference_self_attn_model_1_layer_call_fn_150409��������;�8
1�.
(�%
inputs��������� 
p 
� "W�T
*�'
tensor_0��������� 
&�#
tensor_1���������DD�
$__inference_signature_wrapper_149847�'$�./01?�IJKLZ�efgh���������;�8
� 
1�.
,
input_4!�
input_4���������"]�Z
X
spectral_normalization_17;�8
spectral_normalization_17����������
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_149938z$&�;�8
1�.
(�%
inputs���������
p
� "5�2
+�(
tensor_0����������
� �
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_149961y$�;�8
1�.
(�%
inputs���������
p 
� "5�2
+�(
tensor_0����������
� �
:__inference_spectral_normalization_14_layer_call_fn_149877o$&�;�8
1�.
(�%
inputs���������
p
� "*�'
unknown�����������
:__inference_spectral_normalization_14_layer_call_fn_149886n$�;�8
1�.
(�%
inputs���������
p 
� "*�'
unknown�����������
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_150105z?A�<�9
2�/
)�&
inputs����������
p
� "4�1
*�'
tensor_0���������	@
� �
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_150128y?�<�9
2�/
)�&
inputs����������
p 
� "4�1
*�'
tensor_0���������	@
� �
:__inference_spectral_normalization_15_layer_call_fn_150044o?A�<�9
2�/
)�&
inputs����������
p
� ")�&
unknown���������	@�
:__inference_spectral_normalization_15_layer_call_fn_150053n?�<�9
2�/
)�&
inputs����������
p 
� ")�&
unknown���������	@�
U__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_150272yZ\�;�8
1�.
(�%
inputs���������	@
p
� "4�1
*�'
tensor_0��������� 
� �
U__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_150295xZ�;�8
1�.
(�%
inputs���������	@
p 
� "4�1
*�'
tensor_0��������� 
� �
:__inference_spectral_normalization_16_layer_call_fn_150211nZ\�;�8
1�.
(�%
inputs���������	@
p
� ")�&
unknown��������� �
:__inference_spectral_normalization_16_layer_call_fn_150220mZ�;�8
1�.
(�%
inputs���������	@
p 
� ")�&
unknown��������� �
U__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_150621{���;�8
1�.
(�%
inputs��������� 
p
� "4�1
*�'
tensor_0���������
� �
U__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_150652y��;�8
1�.
(�%
inputs��������� 
p 
� "4�1
*�'
tensor_0���������
� �
:__inference_spectral_normalization_17_layer_call_fn_150552p���;�8
1�.
(�%
inputs��������� 
p
� ")�&
unknown����������
:__inference_spectral_normalization_17_layer_call_fn_150561n��;�8
1�.
(�%
inputs��������� 
p 
� ")�&
unknown���������
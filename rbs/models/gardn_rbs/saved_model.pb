└■#
Ді
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
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
Џ
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
└
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
ч
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
epsilonfloat%иЛ8"&
exponential_avg_factorfloat%  ђ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
є
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ
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
dtypetypeѕ
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
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
┴
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
executor_typestring ѕе
@
StaticRegexFullMatch	
input

output
"
patternstring
э
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
-
Tanh
x"T
y"T"
Ttype:

2
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.10.02unknown8─▀
ћ
spectral_normalization_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name spectral_normalization_15/bias
Ї
2spectral_normalization_15/bias/Read/ReadVariableOpReadVariableOpspectral_normalization_15/bias*
_output_shapes
:*
dtype0
ў
 self_attn_model_1/conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" self_attn_model_1/conv2d_11/bias
Љ
4self_attn_model_1/conv2d_11/bias/Read/ReadVariableOpReadVariableOp self_attn_model_1/conv2d_11/bias*
_output_shapes
:*
dtype0
е
"self_attn_model_1/conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"self_attn_model_1/conv2d_11/kernel
А
6self_attn_model_1/conv2d_11/kernel/Read/ReadVariableOpReadVariableOp"self_attn_model_1/conv2d_11/kernel*&
_output_shapes
:*
dtype0
ў
 self_attn_model_1/conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" self_attn_model_1/conv2d_10/bias
Љ
4self_attn_model_1/conv2d_10/bias/Read/ReadVariableOpReadVariableOp self_attn_model_1/conv2d_10/bias*
_output_shapes
:*
dtype0
е
"self_attn_model_1/conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"self_attn_model_1/conv2d_10/kernel
А
6self_attn_model_1/conv2d_10/kernel/Read/ReadVariableOpReadVariableOp"self_attn_model_1/conv2d_10/kernel*&
_output_shapes
:*
dtype0
ќ
self_attn_model_1/conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!self_attn_model_1/conv2d_9/bias
Ј
3self_attn_model_1/conv2d_9/bias/Read/ReadVariableOpReadVariableOpself_attn_model_1/conv2d_9/bias*
_output_shapes
:*
dtype0
д
!self_attn_model_1/conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!self_attn_model_1/conv2d_9/kernel
Ъ
5self_attn_model_1/conv2d_9/kernel/Read/ReadVariableOpReadVariableOp!self_attn_model_1/conv2d_9/kernel*&
_output_shapes
:*
dtype0
о
Aself_attn_model_1/private__attention_1/private__attention_1_gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *R
shared_nameCAself_attn_model_1/private__attention_1/private__attention_1_gamma
¤
Uself_attn_model_1/private__attention_1/private__attention_1_gamma/Read/ReadVariableOpReadVariableOpAself_attn_model_1/private__attention_1/private__attention_1_gamma*
_output_shapes
: *
dtype0
ћ
spectral_normalization_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name spectral_normalization_14/bias
Ї
2spectral_normalization_14/bias/Read/ReadVariableOpReadVariableOpspectral_normalization_14/bias*
_output_shapes
:*
dtype0
ћ
spectral_normalization_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*/
shared_name spectral_normalization_13/bias
Ї
2spectral_normalization_13/bias/Read/ReadVariableOpReadVariableOpspectral_normalization_13/bias*
_output_shapes
:0*
dtype0
ћ
spectral_normalization_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*/
shared_name spectral_normalization_12/bias
Ї
2spectral_normalization_12/bias/Read/ReadVariableOpReadVariableOpspectral_normalization_12/bias*
_output_shapes
:`*
dtype0
ў
spectral_normalization_15/sn_uVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name spectral_normalization_15/sn_u
Љ
2spectral_normalization_15/sn_u/Read/ReadVariableOpReadVariableOpspectral_normalization_15/sn_u*
_output_shapes

:*
dtype0
ц
 spectral_normalization_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" spectral_normalization_15/kernel
Ю
4spectral_normalization_15/kernel/Read/ReadVariableOpReadVariableOp spectral_normalization_15/kernel*&
_output_shapes
:*
dtype0
б
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_5/moving_variance
Џ
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:*
dtype0
џ
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_5/moving_mean
Њ
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:*
dtype0
ї
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_5/beta
Ё
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:*
dtype0
ј
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_5/gamma
Є
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:*
dtype0
ў
spectral_normalization_14/sn_uVarHandleOp*
_output_shapes
: *
dtype0*
shape
:0*/
shared_name spectral_normalization_14/sn_u
Љ
2spectral_normalization_14/sn_u/Read/ReadVariableOpReadVariableOpspectral_normalization_14/sn_u*
_output_shapes

:0*
dtype0
ц
 spectral_normalization_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*1
shared_name" spectral_normalization_14/kernel
Ю
4spectral_normalization_14/kernel/Read/ReadVariableOpReadVariableOp spectral_normalization_14/kernel*&
_output_shapes
:0*
dtype0
б
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*6
shared_name'%batch_normalization_4/moving_variance
Џ
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:0*
dtype0
џ
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!batch_normalization_4/moving_mean
Њ
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:0*
dtype0
ї
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_namebatch_normalization_4/beta
Ё
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:0*
dtype0
ј
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*,
shared_namebatch_normalization_4/gamma
Є
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:0*
dtype0
ў
spectral_normalization_13/sn_uVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*/
shared_name spectral_normalization_13/sn_u
Љ
2spectral_normalization_13/sn_u/Read/ReadVariableOpReadVariableOpspectral_normalization_13/sn_u*
_output_shapes

:`*
dtype0
ц
 spectral_normalization_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0`*1
shared_name" spectral_normalization_13/kernel
Ю
4spectral_normalization_13/kernel/Read/ReadVariableOpReadVariableOp spectral_normalization_13/kernel*&
_output_shapes
:0`*
dtype0
б
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*6
shared_name'%batch_normalization_3/moving_variance
Џ
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:`*
dtype0
џ
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*2
shared_name#!batch_normalization_3/moving_mean
Њ
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:`*
dtype0
ї
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*+
shared_namebatch_normalization_3/beta
Ё
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:`*
dtype0
ј
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*,
shared_namebatch_normalization_3/gamma
Є
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:`*
dtype0
Ў
spectral_normalization_12/sn_uVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*/
shared_name spectral_normalization_12/sn_u
њ
2spectral_normalization_12/sn_u/Read/ReadVariableOpReadVariableOpspectral_normalization_12/sn_u*
_output_shapes
:	ђ*
dtype0
Ц
 spectral_normalization_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	`ђ*1
shared_name" spectral_normalization_12/kernel
ъ
4spectral_normalization_12/kernel/Read/ReadVariableOpReadVariableOp spectral_normalization_12/kernel*'
_output_shapes
:	`ђ*
dtype0
|
serving_default_input_4Placeholder*(
_output_shapes
:         ђ*
dtype0*
shape:         ђ
Ќ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4 spectral_normalization_12/kernelspectral_normalization_12/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variance spectral_normalization_13/kernelspectral_normalization_13/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance spectral_normalization_14/kernelspectral_normalization_14/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variance!self_attn_model_1/conv2d_9/kernelself_attn_model_1/conv2d_9/bias"self_attn_model_1/conv2d_10/kernel self_attn_model_1/conv2d_10/bias"self_attn_model_1/conv2d_11/kernel self_attn_model_1/conv2d_11/biasAself_attn_model_1/private__attention_1/private__attention_1_gamma spectral_normalization_15/kernelspectral_normalization_15/bias*'
Tin 
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *=
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *-
f(R&
$__inference_signature_wrapper_388130

NoOpNoOp
вѕ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Цѕ
valueџѕBќѕ Bјѕ
К
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
ј
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
└
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
Н
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
ј
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses* 
└
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
Н
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
ј
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses* 
└
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
Н
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
ј
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses* 
╚
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
─
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses
	layer
ђw
Ђw_shape
	ѓsn_u
ѓu*
 
$0
Ѓ1
&2
.3
/4
05
16
?7
ё8
A9
I10
J11
K12
L13
Z14
Ё15
\16
e17
f18
g19
h20
є21
Є22
ѕ23
Ѕ24
і25
І26
ї27
ђ28
Ї29
ѓ30*
«
$0
Ѓ1
.2
/3
?4
ё5
I6
J7
Z8
Ё9
e10
f11
є12
Є13
ѕ14
Ѕ15
і16
І17
ї18
ђ19
Ї20*
* 
х
јnon_trainable_variables
Јlayers
љmetrics
 Љlayer_regularization_losses
њlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
Њtrace_0
ћtrace_1
Ћtrace_2
ќtrace_3* 
:
Ќtrace_0
ўtrace_1
Ўtrace_2
џtrace_3* 
* 

Џserving_default* 
* 
* 
* 
ќ
юnon_trainable_variables
Юlayers
ъmetrics
 Ъlayer_regularization_losses
аlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Аtrace_0* 

бtrace_0* 

$0
Ѓ1
&2*

$0
Ѓ1*
* 
ў
Бnon_trainable_variables
цlayers
Цmetrics
 дlayer_regularization_losses
Дlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

еtrace_0
Еtrace_1* 

фtrace_0
Фtrace_1* 
л
г	variables
Гtrainable_variables
«regularization_losses
»	keras_api
░__call__
+▒&call_and_return_all_conditional_losses

$kernel
	Ѓbias
!▓_jit_compiled_convolution_op*
ke
VARIABLE_VALUE spectral_normalization_12/kernel1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUE*
* 
lf
VARIABLE_VALUEspectral_normalization_12/sn_u4layer_with_weights-0/sn_u/.ATTRIBUTES/VARIABLE_VALUE*
 
.0
/1
02
13*

.0
/1*
* 
ў
│non_trainable_variables
┤layers
хmetrics
 Хlayer_regularization_losses
иlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*

Иtrace_0
╣trace_1* 

║trace_0
╗trace_1* 
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
ќ
╝non_trainable_variables
йlayers
Йmetrics
 ┐layer_regularization_losses
└layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses* 

┴trace_0* 

┬trace_0* 

?0
ё1
A2*

?0
ё1*
* 
ў
├non_trainable_variables
─layers
┼metrics
 кlayer_regularization_losses
Кlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*

╚trace_0
╔trace_1* 

╩trace_0
╦trace_1* 
л
╠	variables
═trainable_variables
╬regularization_losses
¤	keras_api
л__call__
+Л&call_and_return_all_conditional_losses

?kernel
	ёbias
!м_jit_compiled_convolution_op*
ke
VARIABLE_VALUE spectral_normalization_13/kernel1layer_with_weights-2/w/.ATTRIBUTES/VARIABLE_VALUE*
* 
lf
VARIABLE_VALUEspectral_normalization_13/sn_u4layer_with_weights-2/sn_u/.ATTRIBUTES/VARIABLE_VALUE*
 
I0
J1
K2
L3*

I0
J1*
* 
ў
Мnon_trainable_variables
нlayers
Нmetrics
 оlayer_regularization_losses
Оlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

пtrace_0
┘trace_1* 

┌trace_0
█trace_1* 
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
ќ
▄non_trainable_variables
Пlayers
яmetrics
 ▀layer_regularization_losses
Яlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses* 

рtrace_0* 

Рtrace_0* 

Z0
Ё1
\2*

Z0
Ё1*
* 
ў
сnon_trainable_variables
Сlayers
тmetrics
 Тlayer_regularization_losses
уlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

Уtrace_0
жtrace_1* 

Жtrace_0
вtrace_1* 
л
В	variables
ьtrainable_variables
Ьregularization_losses
№	keras_api
­__call__
+ы&call_and_return_all_conditional_losses

Zkernel
	Ёbias
!Ы_jit_compiled_convolution_op*
ke
VARIABLE_VALUE spectral_normalization_14/kernel1layer_with_weights-4/w/.ATTRIBUTES/VARIABLE_VALUE*
* 
lf
VARIABLE_VALUEspectral_normalization_14/sn_u4layer_with_weights-4/sn_u/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
e0
f1
g2
h3*

e0
f1*
* 
ў
зnon_trainable_variables
Зlayers
шmetrics
 Шlayer_regularization_losses
эlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*

Эtrace_0
щtrace_1* 

Щtrace_0
чtrace_1* 
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
ќ
Чnon_trainable_variables
§layers
■metrics
  layer_regularization_losses
ђlayer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses* 

Ђtrace_0* 

ѓtrace_0* 
<
є0
Є1
ѕ2
Ѕ3
і4
І5
ї6*
<
є0
Є1
ѕ2
Ѕ3
і4
І5
ї6*
* 
ў
Ѓnon_trainable_variables
ёlayers
Ёmetrics
 єlayer_regularization_losses
Єlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*

ѕtrace_0
Ѕtrace_1* 

іtrace_0
Іtrace_1* 
├
ї	variables
Їtrainable_variables
јregularization_losses
Ј	keras_api
љ__call__
+Љ&call_and_return_all_conditional_losses
єprivate__attention_1_gamma

єgamma*
Л
њ	variables
Њtrainable_variables
ћregularization_losses
Ћ	keras_api
ќ__call__
+Ќ&call_and_return_all_conditional_losses
Єkernel
	ѕbias
!ў_jit_compiled_convolution_op*
Л
Ў	variables
џtrainable_variables
Џregularization_losses
ю	keras_api
Ю__call__
+ъ&call_and_return_all_conditional_losses
Ѕkernel
	іbias
!Ъ_jit_compiled_convolution_op*
Л
а	variables
Аtrainable_variables
бregularization_losses
Б	keras_api
ц__call__
+Ц&call_and_return_all_conditional_losses
Іkernel
	їbias
!д_jit_compiled_convolution_op*

ђ0
Ї1
ѓ2*

ђ0
Ї1*
* 
ў
Дnon_trainable_variables
еlayers
Еmetrics
 фlayer_regularization_losses
Фlayer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses*

гtrace_0
Гtrace_1* 

«trace_0
»trace_1* 
Л
░	variables
▒trainable_variables
▓regularization_losses
│	keras_api
┤__call__
+х&call_and_return_all_conditional_losses
ђkernel
	Їbias
!Х_jit_compiled_convolution_op*
ke
VARIABLE_VALUE spectral_normalization_15/kernel1layer_with_weights-7/w/.ATTRIBUTES/VARIABLE_VALUE*
* 
lf
VARIABLE_VALUEspectral_normalization_15/sn_u4layer_with_weights-7/sn_u/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEspectral_normalization_12/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEspectral_normalization_13/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEspectral_normalization_14/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
ѓ|
VARIABLE_VALUEAself_attn_model_1/private__attention_1/private__attention_1_gamma'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!self_attn_model_1/conv2d_9/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEself_attn_model_1/conv2d_9/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"self_attn_model_1/conv2d_10/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE self_attn_model_1/conv2d_10/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"self_attn_model_1/conv2d_11/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE self_attn_model_1/conv2d_11/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEspectral_normalization_15/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
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
ѓ9*
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
Ѓ1*

$0
Ѓ1*
* 
ъ
иnon_trainable_variables
Иlayers
╣metrics
 ║layer_regularization_losses
╗layer_metrics
г	variables
Гtrainable_variables
«regularization_losses
░__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses*

╝trace_0* 

йtrace_0* 
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
ё1*

?0
ё1*
* 
ъ
Йnon_trainable_variables
┐layers
└metrics
 ┴layer_regularization_losses
┬layer_metrics
╠	variables
═trainable_variables
╬regularization_losses
л__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses*

├trace_0* 

─trace_0* 
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
Ё1*

Z0
Ё1*
* 
ъ
┼non_trainable_variables
кlayers
Кmetrics
 ╚layer_regularization_losses
╔layer_metrics
В	variables
ьtrainable_variables
Ьregularization_losses
­__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses*

╩trace_0* 

╦trace_0* 
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

є0*

є0*
* 
ъ
╠non_trainable_variables
═layers
╬metrics
 ¤layer_regularization_losses
лlayer_metrics
ї	variables
Їtrainable_variables
јregularization_losses
љ__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses*

Лtrace_0* 

мtrace_0* 

Є0
ѕ1*

Є0
ѕ1*
* 
ъ
Мnon_trainable_variables
нlayers
Нmetrics
 оlayer_regularization_losses
Оlayer_metrics
њ	variables
Њtrainable_variables
ћregularization_losses
ќ__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses*

пtrace_0* 

┘trace_0* 
* 

Ѕ0
і1*

Ѕ0
і1*
* 
ъ
┌non_trainable_variables
█layers
▄metrics
 Пlayer_regularization_losses
яlayer_metrics
Ў	variables
џtrainable_variables
Џregularization_losses
Ю__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses*

▀trace_0* 

Яtrace_0* 
* 

І0
ї1*

І0
ї1*
* 
ъ
рnon_trainable_variables
Рlayers
сmetrics
 Сlayer_regularization_losses
тlayer_metrics
а	variables
Аtrainable_variables
бregularization_losses
ц__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses*

Тtrace_0* 

уtrace_0* 
* 

ѓ0*

0*
* 
* 
* 
* 
* 
* 
* 

ђ0
Ї1*

ђ0
Ї1*
* 
ъ
Уnon_trainable_variables
жlayers
Жmetrics
 вlayer_regularization_losses
Вlayer_metrics
░	variables
▒trainable_variables
▓regularization_losses
┤__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses*

ьtrace_0* 

Ьtrace_0* 
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
═
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename4spectral_normalization_12/kernel/Read/ReadVariableOp2spectral_normalization_12/sn_u/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp4spectral_normalization_13/kernel/Read/ReadVariableOp2spectral_normalization_13/sn_u/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp4spectral_normalization_14/kernel/Read/ReadVariableOp2spectral_normalization_14/sn_u/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp4spectral_normalization_15/kernel/Read/ReadVariableOp2spectral_normalization_15/sn_u/Read/ReadVariableOp2spectral_normalization_12/bias/Read/ReadVariableOp2spectral_normalization_13/bias/Read/ReadVariableOp2spectral_normalization_14/bias/Read/ReadVariableOpUself_attn_model_1/private__attention_1/private__attention_1_gamma/Read/ReadVariableOp5self_attn_model_1/conv2d_9/kernel/Read/ReadVariableOp3self_attn_model_1/conv2d_9/bias/Read/ReadVariableOp6self_attn_model_1/conv2d_10/kernel/Read/ReadVariableOp4self_attn_model_1/conv2d_10/bias/Read/ReadVariableOp6self_attn_model_1/conv2d_11/kernel/Read/ReadVariableOp4self_attn_model_1/conv2d_11/bias/Read/ReadVariableOp2spectral_normalization_15/bias/Read/ReadVariableOpConst*,
Tin%
#2!*
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
GPU2*0J 8ѓ *(
f#R!
__inference__traced_save_389970
▄

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename spectral_normalization_12/kernelspectral_normalization_12/sn_ubatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variance spectral_normalization_13/kernelspectral_normalization_13/sn_ubatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance spectral_normalization_14/kernelspectral_normalization_14/sn_ubatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variance spectral_normalization_15/kernelspectral_normalization_15/sn_uspectral_normalization_12/biasspectral_normalization_13/biasspectral_normalization_14/biasAself_attn_model_1/private__attention_1/private__attention_1_gamma!self_attn_model_1/conv2d_9/kernelself_attn_model_1/conv2d_9/bias"self_attn_model_1/conv2d_10/kernel self_attn_model_1/conv2d_10/bias"self_attn_model_1/conv2d_11/kernel self_attn_model_1/conv2d_11/biasspectral_normalization_15/bias*+
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
GPU2*0J 8ѓ *+
f&R$
"__inference__traced_restore_390073Ќт
п
П
(__inference_model_1_layer_call_fn_387911
input_4"
unknown:	`ђ
	unknown_0:	ђ
	unknown_1:`
	unknown_2:`
	unknown_3:`
	unknown_4:`
	unknown_5:`#
	unknown_6:0`
	unknown_7:`
	unknown_8:0
	unknown_9:0

unknown_10:0

unknown_11:0

unknown_12:0$

unknown_13:0

unknown_14:0

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:$

unknown_20:

unknown_21:$

unknown_22:

unknown_23:$

unknown_24:

unknown_25:

unknown_26: $

unknown_27:

unknown_28:

unknown_29:
identityѕбStatefulPartitionedCallс
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
:         *3
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_387779w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:         ђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_4
Л
a
E__inference_reshape_1_layer_call_and_return_conditional_losses_387073

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Л
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
value	B :R
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :ђЕ
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:         ђa
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
тG
¤
U__inference_spectral_normalization_12_layer_call_and_return_conditional_losses_388888

inputs:
reshape_readvariableop_resource:	`ђD
1spectral_normalize_matmul_readvariableop_resource:	ђ@
2conv2d_transpose_4_biasadd_readvariableop_resource:`
identityѕбReshape/ReadVariableOpб)conv2d_transpose_4/BiasAdd/ReadVariableOpб2conv2d_transpose_4/conv2d_transpose/ReadVariableOpб#spectral_normalize/AssignVariableOpб%spectral_normalize/AssignVariableOp_1б(spectral_normalize/MatMul/ReadVariableOpб!spectral_normalize/ReadVariableOp
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*'
_output_shapes
:	`ђ*
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   u
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0* 
_output_shapes
:
ђђЏ
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0ц
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes
:	ђ*
transpose_b(
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	ђv
%spectral_normalize/l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       └
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
 *╠╝ї+й
'spectral_normalize/l2_normalize/MaximumMaximum,spectral_normalize/l2_normalize/Sum:output:02spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:ё
%spectral_normalize/l2_normalize/RsqrtRsqrt+spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:а
spectral_normalize/l2_normalizeMul#spectral_normalize/MatMul:product:0)spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes
:	ђє
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes
:	ђЃ
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes
:	ђx
'spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       к
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
 *╠╝ї+├
)spectral_normalize/l2_normalize_1/MaximumMaximum.spectral_normalize/l2_normalize_1/Sum:output:04spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:ѕ
'spectral_normalize/l2_normalize_1/RsqrtRsqrt-spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:д
!spectral_normalize/l2_normalize_1Mul%spectral_normalize/MatMul_1:product:0+spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes
:	ђђ
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes
:	ђђ
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	ђЇ
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes
:	ђ▓
spectral_normalize/MatMul_3MatMul%spectral_normalize/MatMul_2:product:0(spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(Э
#spectral_normalize/AssignVariableOpAssignVariableOp1spectral_normalize_matmul_readvariableop_resource(spectral_normalize/StopGradient:output:0)^spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(і
!spectral_normalize/ReadVariableOpReadVariableOpreshape_readvariableop_resource*'
_output_shapes
:	`ђ*
dtype0Е
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*'
_output_shapes
:	`ђy
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   	   `   ђ   б
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*'
_output_shapes
:	`ђш
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(N
conv2d_transpose_4/ShapeShapeinputs*
T0*
_output_shapes
:p
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
valueB:░
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
value	B :\
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :`У
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
valueB:И
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask├
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*'
_output_shapes
:	`ђ*
dtype0Ѓ
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:         `*
paddingSAME*
strides
ў
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0└
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `z
IdentityIdentity#conv2d_transpose_4/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         `П
NoOpNoOp^Reshape/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ђ: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ј
»
:__inference_spectral_normalization_15_layer_call_fn_389477

inputs!
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *^
fYRW
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_387254w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
вU
е
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_389402

inputsA
'conv2d_9_conv2d_readvariableop_resource:6
(conv2d_9_biasadd_readvariableop_resource:B
(conv2d_10_conv2d_readvariableop_resource:7
)conv2d_10_biasadd_readvariableop_resource:B
(conv2d_11_conv2d_readvariableop_resource:7
)conv2d_11_biasadd_readvariableop_resource:<
2private__attention_1_mul_3_readvariableop_resource: 
identity

identity_1ѕб conv2d_10/BiasAdd/ReadVariableOpбconv2d_10/Conv2D/ReadVariableOpб conv2d_11/BiasAdd/ReadVariableOpбconv2d_11/Conv2D/ReadVariableOpбconv2d_9/BiasAdd/ReadVariableOpбconv2d_9/Conv2D/ReadVariableOpб)private__attention_1/Mul_3/ReadVariableOpј
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0г
conv2d_9/Conv2DConv2Dinputs&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
ё
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ў
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         љ
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0«
conv2d_10/Conv2DConv2Dinputs'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
є
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Џ
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         љ
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0«
conv2d_11/Conv2DConv2Dinputs'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
є
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Џ
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         c
private__attention_1/ShapeShapeconv2d_9/BiasAdd:output:0*
T0*
_output_shapes
:r
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
valueB:║
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
valueB:┬
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
valueB:┬
$private__attention_1/strided_slice_2StridedSlice#private__attention_1/Shape:output:03private__attention_1/strided_slice_2/stack:output:05private__attention_1/strided_slice_2/stack_1:output:05private__attention_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskъ
private__attention_1/mulMul-private__attention_1/strided_slice_1:output:0-private__attention_1/strided_slice_2:output:0*
T0*
_output_shapes
: o
$private__attention_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
         м
"private__attention_1/Reshape/shapePack+private__attention_1/strided_slice:output:0private__attention_1/mul:z:0-private__attention_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:«
private__attention_1/ReshapeReshapeconv2d_9/BiasAdd:output:0+private__attention_1/Reshape/shape:output:0*
T0*4
_output_shapes"
 :         D         а
private__attention_1/mul_1Mul-private__attention_1/strided_slice_1:output:0-private__attention_1/strided_slice_2:output:0*
T0*
_output_shapes
: q
&private__attention_1/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
         п
$private__attention_1/Reshape_1/shapePack+private__attention_1/strided_slice:output:0private__attention_1/mul_1:z:0/private__attention_1/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:│
private__attention_1/Reshape_1Reshapeconv2d_10/BiasAdd:output:0-private__attention_1/Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :         D         x
#private__attention_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ┴
private__attention_1/transpose	Transpose'private__attention_1/Reshape_1:output:0,private__attention_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :                  Dа
private__attention_1/mul_2Mul-private__attention_1/strided_slice_1:output:0-private__attention_1/strided_slice_2:output:0*
T0*
_output_shapes
: q
&private__attention_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
         п
$private__attention_1/Reshape_2/shapePack+private__attention_1/strided_slice:output:0private__attention_1/mul_2:z:0/private__attention_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:│
private__attention_1/Reshape_2Reshapeconv2d_11/BiasAdd:output:0-private__attention_1/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :         D         z
%private__attention_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ┼
 private__attention_1/transpose_1	Transpose'private__attention_1/Reshape_2:output:0.private__attention_1/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  DГ
private__attention_1/MatMulBatchMatMulV2%private__attention_1/Reshape:output:0"private__attention_1/transpose:y:0*
T0*+
_output_shapes
:         DDЃ
private__attention_1/SoftmaxSoftmax$private__attention_1/MatMul:output:0*
T0*+
_output_shapes
:         DDz
%private__attention_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ╗
 private__attention_1/transpose_2	Transpose&private__attention_1/Softmax:softmax:0.private__attention_1/transpose_2/perm:output:0*
T0*+
_output_shapes
:         DD╣
private__attention_1/MatMul_1BatchMatMulV2$private__attention_1/transpose_1:y:0$private__attention_1/transpose_2:y:0*
T0*4
_output_shapes"
 :                  Dz
%private__attention_1/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          ─
 private__attention_1/transpose_3	Transpose&private__attention_1/MatMul_1:output:0.private__attention_1/transpose_3/perm:output:0*
T0*4
_output_shapes"
 :         D         q
&private__attention_1/Reshape_3/shape/3Const*
_output_shapes
: *
dtype0*
valueB :
         ќ
$private__attention_1/Reshape_3/shapePack+private__attention_1/strided_slice:output:0-private__attention_1/strided_slice_1:output:0-private__attention_1/strided_slice_2:output:0/private__attention_1/Reshape_3/shape/3:output:0*
N*
T0*
_output_shapes
:┴
private__attention_1/Reshape_3Reshape$private__attention_1/transpose_3:y:0-private__attention_1/Reshape_3/shape:output:0*
T0*8
_output_shapes&
$:"                  ћ
)private__attention_1/Mul_3/ReadVariableOpReadVariableOp2private__attention_1_mul_3_readvariableop_resource*
_output_shapes
: *
dtype0└
private__attention_1/Mul_3Mul'private__attention_1/Reshape_3:output:01private__attention_1/Mul_3/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  Ѓ
private__attention_1/AddAddV2private__attention_1/Mul_3:z:0inputs*
T0*/
_output_shapes
:         s
IdentityIdentityprivate__attention_1/Add:z:0^NoOp*
T0*/
_output_shapes
:         {

Identity_1Identity&private__attention_1/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:         DD┐
NoOpNoOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp*^private__attention_1/Mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : 2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2V
)private__attention_1/Mul_3/ReadVariableOp)private__attention_1/Mul_3/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
вU
е
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_389468

inputsA
'conv2d_9_conv2d_readvariableop_resource:6
(conv2d_9_biasadd_readvariableop_resource:B
(conv2d_10_conv2d_readvariableop_resource:7
)conv2d_10_biasadd_readvariableop_resource:B
(conv2d_11_conv2d_readvariableop_resource:7
)conv2d_11_biasadd_readvariableop_resource:<
2private__attention_1_mul_3_readvariableop_resource: 
identity

identity_1ѕб conv2d_10/BiasAdd/ReadVariableOpбconv2d_10/Conv2D/ReadVariableOpб conv2d_11/BiasAdd/ReadVariableOpбconv2d_11/Conv2D/ReadVariableOpбconv2d_9/BiasAdd/ReadVariableOpбconv2d_9/Conv2D/ReadVariableOpб)private__attention_1/Mul_3/ReadVariableOpј
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0г
conv2d_9/Conv2DConv2Dinputs&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
ё
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ў
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         љ
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0«
conv2d_10/Conv2DConv2Dinputs'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
є
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Џ
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         љ
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0«
conv2d_11/Conv2DConv2Dinputs'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
є
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Џ
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         c
private__attention_1/ShapeShapeconv2d_9/BiasAdd:output:0*
T0*
_output_shapes
:r
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
valueB:║
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
valueB:┬
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
valueB:┬
$private__attention_1/strided_slice_2StridedSlice#private__attention_1/Shape:output:03private__attention_1/strided_slice_2/stack:output:05private__attention_1/strided_slice_2/stack_1:output:05private__attention_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskъ
private__attention_1/mulMul-private__attention_1/strided_slice_1:output:0-private__attention_1/strided_slice_2:output:0*
T0*
_output_shapes
: o
$private__attention_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
         м
"private__attention_1/Reshape/shapePack+private__attention_1/strided_slice:output:0private__attention_1/mul:z:0-private__attention_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:«
private__attention_1/ReshapeReshapeconv2d_9/BiasAdd:output:0+private__attention_1/Reshape/shape:output:0*
T0*4
_output_shapes"
 :         D         а
private__attention_1/mul_1Mul-private__attention_1/strided_slice_1:output:0-private__attention_1/strided_slice_2:output:0*
T0*
_output_shapes
: q
&private__attention_1/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
         п
$private__attention_1/Reshape_1/shapePack+private__attention_1/strided_slice:output:0private__attention_1/mul_1:z:0/private__attention_1/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:│
private__attention_1/Reshape_1Reshapeconv2d_10/BiasAdd:output:0-private__attention_1/Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :         D         x
#private__attention_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ┴
private__attention_1/transpose	Transpose'private__attention_1/Reshape_1:output:0,private__attention_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :                  Dа
private__attention_1/mul_2Mul-private__attention_1/strided_slice_1:output:0-private__attention_1/strided_slice_2:output:0*
T0*
_output_shapes
: q
&private__attention_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
         п
$private__attention_1/Reshape_2/shapePack+private__attention_1/strided_slice:output:0private__attention_1/mul_2:z:0/private__attention_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:│
private__attention_1/Reshape_2Reshapeconv2d_11/BiasAdd:output:0-private__attention_1/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :         D         z
%private__attention_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ┼
 private__attention_1/transpose_1	Transpose'private__attention_1/Reshape_2:output:0.private__attention_1/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  DГ
private__attention_1/MatMulBatchMatMulV2%private__attention_1/Reshape:output:0"private__attention_1/transpose:y:0*
T0*+
_output_shapes
:         DDЃ
private__attention_1/SoftmaxSoftmax$private__attention_1/MatMul:output:0*
T0*+
_output_shapes
:         DDz
%private__attention_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ╗
 private__attention_1/transpose_2	Transpose&private__attention_1/Softmax:softmax:0.private__attention_1/transpose_2/perm:output:0*
T0*+
_output_shapes
:         DD╣
private__attention_1/MatMul_1BatchMatMulV2$private__attention_1/transpose_1:y:0$private__attention_1/transpose_2:y:0*
T0*4
_output_shapes"
 :                  Dz
%private__attention_1/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          ─
 private__attention_1/transpose_3	Transpose&private__attention_1/MatMul_1:output:0.private__attention_1/transpose_3/perm:output:0*
T0*4
_output_shapes"
 :         D         q
&private__attention_1/Reshape_3/shape/3Const*
_output_shapes
: *
dtype0*
valueB :
         ќ
$private__attention_1/Reshape_3/shapePack+private__attention_1/strided_slice:output:0-private__attention_1/strided_slice_1:output:0-private__attention_1/strided_slice_2:output:0/private__attention_1/Reshape_3/shape/3:output:0*
N*
T0*
_output_shapes
:┴
private__attention_1/Reshape_3Reshape$private__attention_1/transpose_3:y:0-private__attention_1/Reshape_3/shape:output:0*
T0*8
_output_shapes&
$:"                  ћ
)private__attention_1/Mul_3/ReadVariableOpReadVariableOp2private__attention_1_mul_3_readvariableop_resource*
_output_shapes
: *
dtype0└
private__attention_1/Mul_3Mul'private__attention_1/Reshape_3:output:01private__attention_1/Mul_3/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  Ѓ
private__attention_1/AddAddV2private__attention_1/Mul_3:z:0inputs*
T0*/
_output_shapes
:         s
IdentityIdentityprivate__attention_1/Add:z:0^NoOp*
T0*/
_output_shapes
:         {

Identity_1Identity&private__attention_1/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:         DD┐
NoOpNoOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp*^private__attention_1/Mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : 2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2V
)private__attention_1/Mul_3/ReadVariableOp)private__attention_1/Mul_3/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
о 
Џ
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_389649

inputsB
(conv2d_transpose_readvariableop_resource:0`-
biasadd_readvariableop_resource:0
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
valueB:┘
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
valueB:┘
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
value	B :0y
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
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskљ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0`*
dtype0▄
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0Ў
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           0y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                           0Ђ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           `: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           `
 
_user_specified_nameinputs
Р/
в
P__inference_private__attention_1_layer_call_and_return_conditional_losses_389754
inputs_0
inputs_1
inputs_2
inputs_3'
mul_3_readvariableop_resource: 
identity

identity_1ѕбMul_3/ReadVariableOp=
ShapeShapeinputs_0*
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
valueB:Л
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
valueB:┘
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
valueB:┘
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
         ~
Reshape/shapePackstrided_slice:output:0mul:z:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:s
ReshapeReshapeinputs_0Reshape/shape:output:0*
T0*4
_output_shapes"
 :         D         a
mul_1Mulstrided_slice_1:output:0strided_slice_2:output:0*
T0*
_output_shapes
: \
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
         ё
Reshape_1/shapePackstrided_slice:output:0	mul_1:z:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:w
	Reshape_1Reshapeinputs_1Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :         D         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ѓ
	transpose	TransposeReshape_1:output:0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  Da
mul_2Mulstrided_slice_1:output:0strided_slice_2:output:0*
T0*
_output_shapes
: \
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
         ё
Reshape_2/shapePackstrided_slice:output:0	mul_2:z:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:w
	Reshape_2Reshapeinputs_2Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :         D         e
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          є
transpose_1	TransposeReshape_2:output:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  Dn
MatMulBatchMatMulV2Reshape:output:0transpose:y:0*
T0*+
_output_shapes
:         DDY
SoftmaxSoftmaxMatMul:output:0*
T0*+
_output_shapes
:         DDe
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_2	TransposeSoftmax:softmax:0transpose_2/perm:output:0*
T0*+
_output_shapes
:         DDz
MatMul_1BatchMatMulV2transpose_1:y:0transpose_2:y:0*
T0*4
_output_shapes"
 :                  De
transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ё
transpose_3	TransposeMatMul_1:output:0transpose_3/perm:output:0*
T0*4
_output_shapes"
 :         D         \
Reshape_3/shape/3Const*
_output_shapes
: *
dtype0*
valueB :
         Г
Reshape_3/shapePackstrided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0Reshape_3/shape/3:output:0*
N*
T0*
_output_shapes
:ѓ
	Reshape_3Reshapetranspose_3:y:0Reshape_3/shape:output:0*
T0*8
_output_shapes&
$:"                  j
Mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes
: *
dtype0Ђ
Mul_3MulReshape_3:output:0Mul_3/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  [
AddAddV2	Mul_3:z:0inputs_3*
T0*/
_output_shapes
:         ^
IdentityIdentityAdd:z:0^NoOp*
T0*/
_output_shapes
:         f

Identity_1IdentitySoftmax:softmax:0^NoOp*
T0*+
_output_shapes
:         DD]
NoOpNoOp^Mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Ђ
_input_shapesp
n:         :         :         :         : 2,
Mul_3/ReadVariableOpMul_3/ReadVariableOp:Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/2:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/3
═
Е
3__inference_conv2d_transpose_4_layer_call_fn_389574

inputs"
unknown:	`ђ
	unknown_0:`
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_386500Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
═
№
U__inference_spectral_normalization_12_layer_call_and_return_conditional_losses_388836

inputsV
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource:	`ђ@
2conv2d_transpose_4_biasadd_readvariableop_resource:`
identityѕб)conv2d_transpose_4/BiasAdd/ReadVariableOpб2conv2d_transpose_4/conv2d_transpose/ReadVariableOpN
conv2d_transpose_4/ShapeShapeinputs*
T0*
_output_shapes
:p
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
valueB:░
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
value	B :\
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :`У
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
valueB:И
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskи
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*'
_output_shapes
:	`ђ*
dtype0Ѓ
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:         `*
paddingSAME*
strides
ў
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0└
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `z
IdentityIdentity#conv2d_transpose_4/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         `Д
NoOpNoOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
у
_
C__inference_re_lu_5_layer_call_and_return_conditional_losses_387212

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
тG
¤
U__inference_spectral_normalization_12_layer_call_and_return_conditional_losses_387620

inputs:
reshape_readvariableop_resource:	`ђD
1spectral_normalize_matmul_readvariableop_resource:	ђ@
2conv2d_transpose_4_biasadd_readvariableop_resource:`
identityѕбReshape/ReadVariableOpб)conv2d_transpose_4/BiasAdd/ReadVariableOpб2conv2d_transpose_4/conv2d_transpose/ReadVariableOpб#spectral_normalize/AssignVariableOpб%spectral_normalize/AssignVariableOp_1б(spectral_normalize/MatMul/ReadVariableOpб!spectral_normalize/ReadVariableOp
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*'
_output_shapes
:	`ђ*
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   u
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0* 
_output_shapes
:
ђђЏ
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0ц
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes
:	ђ*
transpose_b(
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	ђv
%spectral_normalize/l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       └
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
 *╠╝ї+й
'spectral_normalize/l2_normalize/MaximumMaximum,spectral_normalize/l2_normalize/Sum:output:02spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:ё
%spectral_normalize/l2_normalize/RsqrtRsqrt+spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:а
spectral_normalize/l2_normalizeMul#spectral_normalize/MatMul:product:0)spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes
:	ђє
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes
:	ђЃ
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes
:	ђx
'spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       к
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
 *╠╝ї+├
)spectral_normalize/l2_normalize_1/MaximumMaximum.spectral_normalize/l2_normalize_1/Sum:output:04spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:ѕ
'spectral_normalize/l2_normalize_1/RsqrtRsqrt-spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:д
!spectral_normalize/l2_normalize_1Mul%spectral_normalize/MatMul_1:product:0+spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes
:	ђђ
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes
:	ђђ
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	ђЇ
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes
:	ђ▓
spectral_normalize/MatMul_3MatMul%spectral_normalize/MatMul_2:product:0(spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(Э
#spectral_normalize/AssignVariableOpAssignVariableOp1spectral_normalize_matmul_readvariableop_resource(spectral_normalize/StopGradient:output:0)^spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(і
!spectral_normalize/ReadVariableOpReadVariableOpreshape_readvariableop_resource*'
_output_shapes
:	`ђ*
dtype0Е
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*'
_output_shapes
:	`ђy
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   	   `   ђ   б
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*'
_output_shapes
:	`ђш
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(N
conv2d_transpose_4/ShapeShapeinputs*
T0*
_output_shapes
:p
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
valueB:░
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
value	B :\
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :`У
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
valueB:И
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask├
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*'
_output_shapes
:	`ђ*
dtype0Ѓ
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:         `*
paddingSAME*
strides
ў
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0└
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `z
IdentityIdentity#conv2d_transpose_4/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         `П
NoOpNoOp^Reshape/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ђ: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
ћ	
Л
6__inference_batch_normalization_3_layer_call_fn_388901

inputs
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identityѕбStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           `*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_386529Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           `: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           `
 
_user_specified_nameinputs
у
_
C__inference_re_lu_3_layer_call_and_return_conditional_losses_388960

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         `b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         `:W S
/
_output_shapes
:         `
 
_user_specified_nameinputs
Е

■
E__inference_conv2d_10_layer_call_and_return_conditional_losses_386820

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
њ
░
:__inference_spectral_normalization_12_layer_call_fn_388802

inputs"
unknown:	`ђ
	unknown_0:`
identityѕбStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *^
fYRW
U__inference_spectral_normalization_12_layer_call_and_return_conditional_losses_387098w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
є
└
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_386560

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           `:`:`:`:`:*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           `н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           `: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           `
 
_user_specified_nameinputs
є
└
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_389117

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           0:0:0:0:0:*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           0н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           0
 
_user_specified_nameinputs
є
└
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_389284

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╔
Ь
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_389170

inputsU
;conv2d_transpose_6_conv2d_transpose_readvariableop_resource:0@
2conv2d_transpose_6_biasadd_readvariableop_resource:
identityѕб)conv2d_transpose_6/BiasAdd/ReadVariableOpб2conv2d_transpose_6/conv2d_transpose/ReadVariableOpN
conv2d_transpose_6/ShapeShapeinputs*
T0*
_output_shapes
:p
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
valueB:░
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
value	B :У
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
valueB:И
"conv2d_transpose_6/strided_slice_1StridedSlice!conv2d_transpose_6/stack:output:01conv2d_transpose_6/strided_slice_1/stack:output:03conv2d_transpose_6/strided_slice_1/stack_1:output:03conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskХ
2conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0*
dtype0Ѓ
#conv2d_transpose_6/conv2d_transposeConv2DBackpropInput!conv2d_transpose_6/stack:output:0:conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ў
)conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0└
conv2d_transpose_6/BiasAddBiasAdd,conv2d_transpose_6/conv2d_transpose:output:01conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         z
IdentityIdentity#conv2d_transpose_6/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         Д
NoOpNoOp*^conv2d_transpose_6/BiasAdd/ReadVariableOp3^conv2d_transpose_6/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	0: : 2V
)conv2d_transpose_6/BiasAdd/ReadVariableOp)conv2d_transpose_6/BiasAdd/ReadVariableOp2h
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2conv2d_transpose_6/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:         	0
 
_user_specified_nameinputs
╣
╠
:__inference_spectral_normalization_14_layer_call_fn_389147

inputs!
unknown:0
	unknown_0:0
	unknown_1:
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *^
fYRW
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_387462w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         	0: : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         	0
 
_user_specified_nameinputs
Ю
╚
2__inference_self_attn_model_1_layer_call_fn_389336

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
	unknown_5: 
identity

identity_1ѕбStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:         :         DD*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_386988w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         u

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*+
_output_shapes
:         DD`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
е

§
D__inference_conv2d_9_layer_call_and_return_conditional_losses_386804

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
о 
Џ
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_389691

inputsB
(conv2d_transpose_readvariableop_resource:0-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
valueB:┘
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
valueB:┘
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
value	B :y
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
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskљ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0*
dtype0▄
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                           Ђ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           0
 
_user_specified_nameinputs
Е

■
E__inference_conv2d_11_layer_call_and_return_conditional_losses_386836

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
о 
Џ
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_386608

inputsB
(conv2d_transpose_readvariableop_resource:0`-
biasadd_readvariableop_resource:0
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
valueB:┘
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
valueB:┘
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
value	B :0y
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
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskљ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0`*
dtype0▄
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0Ў
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           0y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                           0Ђ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           `: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           `
 
_user_specified_nameinputs
њ	
Л
6__inference_batch_normalization_4_layer_call_fn_389081

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identityѕбStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_386668Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           0
 
_user_specified_nameinputs
╩
е
3__inference_conv2d_transpose_5_layer_call_fn_389616

inputs!
unknown:0`
	unknown_0:0
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_386608Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           `: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           `
 
_user_specified_nameinputs
№
Ъ
*__inference_conv2d_10_layer_call_fn_389782

inputs!
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_386820w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
є
└
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_386776

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Й
D
(__inference_re_lu_5_layer_call_fn_389289

inputs
identity╣
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_re_lu_5_layer_call_and_return_conditional_losses_387212h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ю
╚
2__inference_self_attn_model_1_layer_call_fn_389315

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
	unknown_5: 
identity

identity_1ѕбStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:         :         DD*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_386900w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         u

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*+
_output_shapes
:         DD`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
кH
═
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_389565

inputs9
reshape_readvariableop_resource:C
1spectral_normalize_matmul_readvariableop_resource:@
2conv2d_transpose_7_biasadd_readvariableop_resource:
identityѕбReshape/ReadVariableOpб)conv2d_transpose_7/BiasAdd/ReadVariableOpб2conv2d_transpose_7/conv2d_transpose/ReadVariableOpб#spectral_normalize/AssignVariableOpб%spectral_normalize/AssignVariableOp_1б(spectral_normalize/MatMul/ReadVariableOpб!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:*
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       s
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes

:џ
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Б
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
valueB"       └
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
 *╠╝ї+й
'spectral_normalize/l2_normalize/MaximumMaximum,spectral_normalize/l2_normalize/Sum:output:02spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:ё
%spectral_normalize/l2_normalize/RsqrtRsqrt+spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:Ъ
spectral_normalize/l2_normalizeMul#spectral_normalize/MatMul:product:0)spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes

:Ё
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes

:ѓ
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:x
'spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       к
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
 *╠╝ї+├
)spectral_normalize/l2_normalize_1/MaximumMaximum.spectral_normalize/l2_normalize_1/Sum:output:04spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:ѕ
'spectral_normalize/l2_normalize_1/RsqrtRsqrt-spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:Ц
!spectral_normalize/l2_normalize_1Mul%spectral_normalize/MatMul_1:product:0+spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

:
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes

:ї
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes

:▓
spectral_normalize/MatMul_3MatMul%spectral_normalize/MatMul_2:product:0(spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(Э
#spectral_normalize/AssignVariableOpAssignVariableOp1spectral_normalize_matmul_readvariableop_resource(spectral_normalize/StopGradient:output:0)^spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ѕ
!spectral_normalize/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:*
dtype0е
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            А
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:ш
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(N
conv2d_transpose_7/ShapeShapeinputs*
T0*
_output_shapes
:p
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
valueB:░
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
value	B :У
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
valueB:И
"conv2d_transpose_7/strided_slice_1StridedSlice!conv2d_transpose_7/stack:output:01conv2d_transpose_7/strided_slice_1/stack:output:03conv2d_transpose_7/strided_slice_1/stack_1:output:03conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask┬
2conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
:*
dtype0Ѓ
#conv2d_transpose_7/conv2d_transposeConv2DBackpropInput!conv2d_transpose_7/stack:output:0:conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ў
)conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0└
conv2d_transpose_7/BiasAddBiasAdd,conv2d_transpose_7/conv2d_transpose:output:01conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ~
conv2d_transpose_7/TanhTanh#conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:         r
IdentityIdentityconv2d_transpose_7/Tanh:y:0^NoOp*
T0*/
_output_shapes
:         П
NoOpNoOp^Reshape/ReadVariableOp*^conv2d_transpose_7/BiasAdd/ReadVariableOp3^conv2d_transpose_7/conv2d_transpose/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2V
)conv2d_transpose_7/BiasAdd/ReadVariableOp)conv2d_transpose_7/BiasAdd/ReadVariableOp2h
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
њ	
Л
6__inference_batch_normalization_5_layer_call_fn_389248

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_386776Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ћ	
Л
6__inference_batch_normalization_4_layer_call_fn_389068

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identityѕбStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_386637Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           0
 
_user_specified_nameinputs
ь
ъ
)__inference_conv2d_9_layer_call_fn_389763

inputs!
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_386804w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
у
_
C__inference_re_lu_4_layer_call_and_return_conditional_losses_389127

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         	0b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         	0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         	0:W S
/
_output_shapes
:         	0
 
_user_specified_nameinputs
ф
с
(__inference_model_1_layer_call_fn_388189

inputs"
unknown:	`ђ
	unknown_0:`
	unknown_1:`
	unknown_2:`
	unknown_3:`
	unknown_4:`#
	unknown_5:0`
	unknown_6:0
	unknown_7:0
	unknown_8:0
	unknown_9:0

unknown_10:0$

unknown_11:0

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17:

unknown_18:$

unknown_19:

unknown_20:$

unknown_21:

unknown_22:

unknown_23: $

unknown_24:

unknown_25:
identityѕбStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:         *=
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_387261w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:         ђ: : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
║!
Џ
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_389854

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
valueB:┘
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
valueB:┘
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
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskљ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0▄
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           j
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           q
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+                           Ђ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╬G
═
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_389222

inputs9
reshape_readvariableop_resource:0C
1spectral_normalize_matmul_readvariableop_resource:0@
2conv2d_transpose_6_biasadd_readvariableop_resource:
identityѕбReshape/ReadVariableOpб)conv2d_transpose_6/BiasAdd/ReadVariableOpб2conv2d_transpose_6/conv2d_transpose/ReadVariableOpб#spectral_normalize/AssignVariableOpб%spectral_normalize/AssignVariableOp_1б(spectral_normalize/MatMul/ReadVariableOpб!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:0*
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    0   s
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes

:H0џ
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:0*
dtype0Б
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
valueB"       └
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
 *╠╝ї+й
'spectral_normalize/l2_normalize/MaximumMaximum,spectral_normalize/l2_normalize/Sum:output:02spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:ё
%spectral_normalize/l2_normalize/RsqrtRsqrt+spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:Ъ
spectral_normalize/l2_normalizeMul#spectral_normalize/MatMul:product:0)spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes

:HЁ
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes

:0ѓ
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:0x
'spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       к
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
 *╠╝ї+├
)spectral_normalize/l2_normalize_1/MaximumMaximum.spectral_normalize/l2_normalize_1/Sum:output:04spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:ѕ
'spectral_normalize/l2_normalize_1/RsqrtRsqrt-spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:Ц
!spectral_normalize/l2_normalize_1Mul%spectral_normalize/MatMul_1:product:0+spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

:0
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:0
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes

:Hї
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes

:0▓
spectral_normalize/MatMul_3MatMul%spectral_normalize/MatMul_2:product:0(spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(Э
#spectral_normalize/AssignVariableOpAssignVariableOp1spectral_normalize_matmul_readvariableop_resource(spectral_normalize/StopGradient:output:0)^spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ѕ
!spectral_normalize/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:0*
dtype0е
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:0y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         0   А
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:0ш
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(N
conv2d_transpose_6/ShapeShapeinputs*
T0*
_output_shapes
:p
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
valueB:░
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
value	B :У
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
valueB:И
"conv2d_transpose_6/strided_slice_1StridedSlice!conv2d_transpose_6/stack:output:01conv2d_transpose_6/strided_slice_1/stack:output:03conv2d_transpose_6/strided_slice_1/stack_1:output:03conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask┬
2conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
:0*
dtype0Ѓ
#conv2d_transpose_6/conv2d_transposeConv2DBackpropInput!conv2d_transpose_6/stack:output:0:conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ў
)conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0└
conv2d_transpose_6/BiasAddBiasAdd,conv2d_transpose_6/conv2d_transpose:output:01conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         z
IdentityIdentity#conv2d_transpose_6/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         П
NoOpNoOp^Reshape/ReadVariableOp*^conv2d_transpose_6/BiasAdd/ReadVariableOp3^conv2d_transpose_6/conv2d_transpose/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         	0: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2V
)conv2d_transpose_6/BiasAdd/ReadVariableOp)conv2d_transpose_6/BiasAdd/ReadVariableOp2h
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:         	0
 
_user_specified_nameinputs
Ј
»
:__inference_spectral_normalization_14_layer_call_fn_389136

inputs!
unknown:0
	unknown_0:
identityѕбStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *^
fYRW
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_387188w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	0: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         	0
 
_user_specified_nameinputs
њ	
Л
6__inference_batch_normalization_3_layer_call_fn_388914

inputs
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identityѕбStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_386560Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           `: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           `
 
_user_specified_nameinputs
кH
═
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_387383

inputs9
reshape_readvariableop_resource:C
1spectral_normalize_matmul_readvariableop_resource:@
2conv2d_transpose_7_biasadd_readvariableop_resource:
identityѕбReshape/ReadVariableOpб)conv2d_transpose_7/BiasAdd/ReadVariableOpб2conv2d_transpose_7/conv2d_transpose/ReadVariableOpб#spectral_normalize/AssignVariableOpб%spectral_normalize/AssignVariableOp_1б(spectral_normalize/MatMul/ReadVariableOpб!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:*
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       s
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes

:џ
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Б
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
valueB"       └
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
 *╠╝ї+й
'spectral_normalize/l2_normalize/MaximumMaximum,spectral_normalize/l2_normalize/Sum:output:02spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:ё
%spectral_normalize/l2_normalize/RsqrtRsqrt+spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:Ъ
spectral_normalize/l2_normalizeMul#spectral_normalize/MatMul:product:0)spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes

:Ё
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes

:ѓ
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:x
'spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       к
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
 *╠╝ї+├
)spectral_normalize/l2_normalize_1/MaximumMaximum.spectral_normalize/l2_normalize_1/Sum:output:04spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:ѕ
'spectral_normalize/l2_normalize_1/RsqrtRsqrt-spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:Ц
!spectral_normalize/l2_normalize_1Mul%spectral_normalize/MatMul_1:product:0+spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

:
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes

:ї
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes

:▓
spectral_normalize/MatMul_3MatMul%spectral_normalize/MatMul_2:product:0(spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(Э
#spectral_normalize/AssignVariableOpAssignVariableOp1spectral_normalize_matmul_readvariableop_resource(spectral_normalize/StopGradient:output:0)^spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ѕ
!spectral_normalize/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:*
dtype0е
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            А
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:ш
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(N
conv2d_transpose_7/ShapeShapeinputs*
T0*
_output_shapes
:p
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
valueB:░
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
value	B :У
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
valueB:И
"conv2d_transpose_7/strided_slice_1StridedSlice!conv2d_transpose_7/stack:output:01conv2d_transpose_7/strided_slice_1/stack:output:03conv2d_transpose_7/strided_slice_1/stack_1:output:03conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask┬
2conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
:*
dtype0Ѓ
#conv2d_transpose_7/conv2d_transposeConv2DBackpropInput!conv2d_transpose_7/stack:output:0:conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ў
)conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0└
conv2d_transpose_7/BiasAddBiasAdd,conv2d_transpose_7/conv2d_transpose:output:01conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ~
conv2d_transpose_7/TanhTanh#conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:         r
IdentityIdentityconv2d_transpose_7/Tanh:y:0^NoOp*
T0*/
_output_shapes
:         П
NoOpNoOp^Reshape/ReadVariableOp*^conv2d_transpose_7/BiasAdd/ReadVariableOp3^conv2d_transpose_7/conv2d_transpose/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2V
)conv2d_transpose_7/BiasAdd/ReadVariableOp)conv2d_transpose_7/BiasAdd/ReadVariableOp2h
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╔
Ь
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_387188

inputsU
;conv2d_transpose_6_conv2d_transpose_readvariableop_resource:0@
2conv2d_transpose_6_biasadd_readvariableop_resource:
identityѕб)conv2d_transpose_6/BiasAdd/ReadVariableOpб2conv2d_transpose_6/conv2d_transpose/ReadVariableOpN
conv2d_transpose_6/ShapeShapeinputs*
T0*
_output_shapes
:p
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
valueB:░
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
value	B :У
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
valueB:И
"conv2d_transpose_6/strided_slice_1StridedSlice!conv2d_transpose_6/stack:output:01conv2d_transpose_6/strided_slice_1/stack:output:03conv2d_transpose_6/strided_slice_1/stack_1:output:03conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskХ
2conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0*
dtype0Ѓ
#conv2d_transpose_6/conv2d_transposeConv2DBackpropInput!conv2d_transpose_6/stack:output:0:conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ў
)conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0└
conv2d_transpose_6/BiasAddBiasAdd,conv2d_transpose_6/conv2d_transpose:output:01conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         z
IdentityIdentity#conv2d_transpose_6/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         Д
NoOpNoOp*^conv2d_transpose_6/BiasAdd/ReadVariableOp3^conv2d_transpose_6/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         	0: : 2V
)conv2d_transpose_6/BiasAdd/ReadVariableOp)conv2d_transpose_6/BiasAdd/ReadVariableOp2h
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2conv2d_transpose_6/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:         	0
 
_user_specified_nameinputs
й
╬
:__inference_spectral_normalization_12_layer_call_fn_388813

inputs"
unknown:	`ђ
	unknown_0:	ђ
	unknown_1:`
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *^
fYRW
U__inference_spectral_normalization_12_layer_call_and_return_conditional_losses_387620w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":         ђ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
║!
Џ
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_387045

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
valueB:┘
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
valueB:┘
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
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskљ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0▄
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           j
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           q
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+                           Ђ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╔
Ь
U__inference_spectral_normalization_13_layer_call_and_return_conditional_losses_389003

inputsU
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource:0`@
2conv2d_transpose_5_biasadd_readvariableop_resource:0
identityѕб)conv2d_transpose_5/BiasAdd/ReadVariableOpб2conv2d_transpose_5/conv2d_transpose/ReadVariableOpN
conv2d_transpose_5/ShapeShapeinputs*
T0*
_output_shapes
:p
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
valueB:░
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
value	B :0У
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
valueB:И
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskХ
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0`*
dtype0Ѓ
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:         	0*
paddingSAME*
strides
ў
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0└
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         	0z
IdentityIdentity#conv2d_transpose_5/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         	0Д
NoOpNoOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         `: : 2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:         `
 
_user_specified_nameinputs
╠
ю
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_386745

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
у
_
C__inference_re_lu_4_layer_call_and_return_conditional_losses_387163

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         	0b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         	0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         	0:W S
/
_output_shapes
:         	0
 
_user_specified_nameinputs
ћ	
Л
6__inference_batch_normalization_5_layer_call_fn_389235

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_386745Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
нG
═
U__inference_spectral_normalization_13_layer_call_and_return_conditional_losses_387541

inputs9
reshape_readvariableop_resource:0`C
1spectral_normalize_matmul_readvariableop_resource:`@
2conv2d_transpose_5_biasadd_readvariableop_resource:0
identityѕбReshape/ReadVariableOpб)conv2d_transpose_5/BiasAdd/ReadVariableOpб2conv2d_transpose_5/conv2d_transpose/ReadVariableOpб#spectral_normalize/AssignVariableOpб%spectral_normalize/AssignVariableOp_1б(spectral_normalize/MatMul/ReadVariableOpб!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:0`*
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   t
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	Я`џ
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:`*
dtype0ц
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes
:	Я*
transpose_b(
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	Яv
%spectral_normalize/l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       └
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
 *╠╝ї+й
'spectral_normalize/l2_normalize/MaximumMaximum,spectral_normalize/l2_normalize/Sum:output:02spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:ё
%spectral_normalize/l2_normalize/RsqrtRsqrt+spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:а
spectral_normalize/l2_normalizeMul#spectral_normalize/MatMul:product:0)spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes
:	ЯЁ
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes

:`ѓ
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:`x
'spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       к
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
 *╠╝ї+├
)spectral_normalize/l2_normalize_1/MaximumMaximum.spectral_normalize/l2_normalize_1/Sum:output:04spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:ѕ
'spectral_normalize/l2_normalize_1/RsqrtRsqrt-spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:Ц
!spectral_normalize/l2_normalize_1Mul%spectral_normalize/MatMul_1:product:0+spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

:`
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:`ђ
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	Яї
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes

:`▓
spectral_normalize/MatMul_3MatMul%spectral_normalize/MatMul_2:product:0(spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(Э
#spectral_normalize/AssignVariableOpAssignVariableOp1spectral_normalize_matmul_readvariableop_resource(spectral_normalize/StopGradient:output:0)^spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ѕ
!spectral_normalize/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:0`*
dtype0е
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:0`y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      0   `   А
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:0`ш
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(N
conv2d_transpose_5/ShapeShapeinputs*
T0*
_output_shapes
:p
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
valueB:░
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
value	B :0У
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
valueB:И
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask┬
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
:0`*
dtype0Ѓ
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:         	0*
paddingSAME*
strides
ў
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0└
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         	0z
IdentityIdentity#conv2d_transpose_5/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         	0П
NoOpNoOp^Reshape/ReadVariableOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         `: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:         `
 
_user_specified_nameinputs
ѕM
у
C__inference_model_1_layer_call_and_return_conditional_losses_387261

inputs;
 spectral_normalization_12_387099:	`ђ.
 spectral_normalization_12_387101:`*
batch_normalization_3_387104:`*
batch_normalization_3_387106:`*
batch_normalization_3_387108:`*
batch_normalization_3_387110:`:
 spectral_normalization_13_387144:0`.
 spectral_normalization_13_387146:0*
batch_normalization_4_387149:0*
batch_normalization_4_387151:0*
batch_normalization_4_387153:0*
batch_normalization_4_387155:0:
 spectral_normalization_14_387189:0.
 spectral_normalization_14_387191:*
batch_normalization_5_387198:*
batch_normalization_5_387200:*
batch_normalization_5_387202:*
batch_normalization_5_387204:2
self_attn_model_1_387214:&
self_attn_model_1_387216:2
self_attn_model_1_387218:&
self_attn_model_1_387220:2
self_attn_model_1_387222:&
self_attn_model_1_387224:"
self_attn_model_1_387226: :
 spectral_normalization_15_387255:.
 spectral_normalization_15_387257:
identityѕб-batch_normalization_3/StatefulPartitionedCallб-batch_normalization_4/StatefulPartitionedCallб-batch_normalization_5/StatefulPartitionedCallб)self_attn_model_1/StatefulPartitionedCallб1spectral_normalization_12/StatefulPartitionedCallб1spectral_normalization_13/StatefulPartitionedCallб1spectral_normalization_14/StatefulPartitionedCallб1spectral_normalization_15/StatefulPartitionedCallк
reshape_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_387073█
1spectral_normalization_12/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0 spectral_normalization_12_387099 spectral_normalization_12_387101*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *^
fYRW
U__inference_spectral_normalization_12_layer_call_and_return_conditional_losses_387098Б
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_12/StatefulPartitionedCall:output:0batch_normalization_3_387104batch_normalization_3_387106batch_normalization_3_387108batch_normalization_3_387110*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_386529ы
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_387118┘
1spectral_normalization_13/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0 spectral_normalization_13_387144 spectral_normalization_13_387146*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *^
fYRW
U__inference_spectral_normalization_13_layer_call_and_return_conditional_losses_387143Б
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_13/StatefulPartitionedCall:output:0batch_normalization_4_387149batch_normalization_4_387151batch_normalization_4_387153batch_normalization_4_387155*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_386637ы
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_re_lu_4_layer_call_and_return_conditional_losses_387163┘
1spectral_normalization_14/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0 spectral_normalization_14_387189 spectral_normalization_14_387191*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *^
fYRW
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_387188Є
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                Ѕ
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               Ѕ
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ї
(tf.__operators__.getitem_1/strided_sliceStridedSlice:spectral_normalization_14/StatefulPartitionedCall:output:07tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_maskџ
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall1tf.__operators__.getitem_1/strided_slice:output:0batch_normalization_5_387198batch_normalization_5_387200batch_normalization_5_387202batch_normalization_5_387204*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_386745ы
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_re_lu_5_layer_call_and_return_conditional_losses_387212П
)self_attn_model_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0self_attn_model_1_387214self_attn_model_1_387216self_attn_model_1_387218self_attn_model_1_387220self_attn_model_1_387222self_attn_model_1_387224self_attn_model_1_387226*
Tin

2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:         :         DD*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_386900в
1spectral_normalization_15/StatefulPartitionedCallStatefulPartitionedCall2self_attn_model_1/StatefulPartitionedCall:output:0 spectral_normalization_15_387255 spectral_normalization_15_387257*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *^
fYRW
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_387254Љ
IdentityIdentity:spectral_normalization_15/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         м
NoOpNoOp.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall*^self_attn_model_1/StatefulPartitionedCall2^spectral_normalization_12/StatefulPartitionedCall2^spectral_normalization_13/StatefulPartitionedCall2^spectral_normalization_14/StatefulPartitionedCall2^spectral_normalization_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:         ђ: : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2V
)self_attn_model_1/StatefulPartitionedCall)self_attn_model_1/StatefulPartitionedCall2f
1spectral_normalization_12/StatefulPartitionedCall1spectral_normalization_12/StatefulPartitionedCall2f
1spectral_normalization_13/StatefulPartitionedCall1spectral_normalization_13/StatefulPartitionedCall2f
1spectral_normalization_14/StatefulPartitionedCall1spectral_normalization_14/StatefulPartitionedCall2f
1spectral_normalization_15/StatefulPartitionedCall1spectral_normalization_15/StatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
є
└
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_388950

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           `:`:`:`:`:*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           `н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           `: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           `
 
_user_specified_nameinputs
╠
ю
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_389099

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           0:0:0:0:0:*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           0░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           0
 
_user_specified_nameinputs
о 
Џ
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_386716

inputsB
(conv2d_transpose_readvariableop_resource:0-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
valueB:┘
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
valueB:┘
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
value	B :y
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
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskљ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0*
dtype0▄
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                           Ђ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           0
 
_user_specified_nameinputs
№
Ъ
*__inference_conv2d_11_layer_call_fn_389801

inputs!
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_386836w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╔
Ь
U__inference_spectral_normalization_13_layer_call_and_return_conditional_losses_387143

inputsU
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource:0`@
2conv2d_transpose_5_biasadd_readvariableop_resource:0
identityѕб)conv2d_transpose_5/BiasAdd/ReadVariableOpб2conv2d_transpose_5/conv2d_transpose/ReadVariableOpN
conv2d_transpose_5/ShapeShapeinputs*
T0*
_output_shapes
:p
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
valueB:░
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
value	B :0У
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
valueB:И
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskХ
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0`*
dtype0Ѓ
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:         	0*
paddingSAME*
strides
ў
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0└
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         	0z
IdentityIdentity#conv2d_transpose_5/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         	0Д
NoOpNoOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         `: : 2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:         `
 
_user_specified_nameinputs
Л
a
E__inference_reshape_1_layer_call_and_return_conditional_losses_388793

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Л
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
value	B :R
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :ђЕ
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:         ђa
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Г
С
(__inference_model_1_layer_call_fn_387318
input_4"
unknown:	`ђ
	unknown_0:`
	unknown_1:`
	unknown_2:`
	unknown_3:`
	unknown_4:`#
	unknown_5:0`
	unknown_6:0
	unknown_7:0
	unknown_8:0
	unknown_9:0

unknown_10:0$

unknown_11:0

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17:

unknown_18:$

unknown_19:

unknown_20:$

unknown_21:

unknown_22:

unknown_23: $

unknown_24:

unknown_25:
identityѕбStatefulPartitionedCall╣
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
:         *=
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_387261w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:         ђ: : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_4
╠
ю
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_386529

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           `:`:`:`:`:*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           `░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           `: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           `
 
_user_specified_nameinputs
ТO
╣
C__inference_model_1_layer_call_and_return_conditional_losses_388069
input_4;
 spectral_normalization_12_387990:	`ђ3
 spectral_normalization_12_387992:	ђ.
 spectral_normalization_12_387994:`*
batch_normalization_3_387997:`*
batch_normalization_3_387999:`*
batch_normalization_3_388001:`*
batch_normalization_3_388003:`:
 spectral_normalization_13_388007:0`2
 spectral_normalization_13_388009:`.
 spectral_normalization_13_388011:0*
batch_normalization_4_388014:0*
batch_normalization_4_388016:0*
batch_normalization_4_388018:0*
batch_normalization_4_388020:0:
 spectral_normalization_14_388024:02
 spectral_normalization_14_388026:0.
 spectral_normalization_14_388028:*
batch_normalization_5_388035:*
batch_normalization_5_388037:*
batch_normalization_5_388039:*
batch_normalization_5_388041:2
self_attn_model_1_388045:&
self_attn_model_1_388047:2
self_attn_model_1_388049:&
self_attn_model_1_388051:2
self_attn_model_1_388053:&
self_attn_model_1_388055:"
self_attn_model_1_388057: :
 spectral_normalization_15_388061:2
 spectral_normalization_15_388063:.
 spectral_normalization_15_388065:
identityѕб-batch_normalization_3/StatefulPartitionedCallб-batch_normalization_4/StatefulPartitionedCallб-batch_normalization_5/StatefulPartitionedCallб)self_attn_model_1/StatefulPartitionedCallб1spectral_normalization_12/StatefulPartitionedCallб1spectral_normalization_13/StatefulPartitionedCallб1spectral_normalization_14/StatefulPartitionedCallб1spectral_normalization_15/StatefulPartitionedCallК
reshape_1/PartitionedCallPartitionedCallinput_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_387073§
1spectral_normalization_12/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0 spectral_normalization_12_387990 spectral_normalization_12_387992 spectral_normalization_12_387994*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *^
fYRW
U__inference_spectral_normalization_12_layer_call_and_return_conditional_losses_387620А
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_12/StatefulPartitionedCall:output:0batch_normalization_3_387997batch_normalization_3_387999batch_normalization_3_388001batch_normalization_3_388003*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_386560ы
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_387118ч
1spectral_normalization_13/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0 spectral_normalization_13_388007 spectral_normalization_13_388009 spectral_normalization_13_388011*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	0*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *^
fYRW
U__inference_spectral_normalization_13_layer_call_and_return_conditional_losses_387541А
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_13/StatefulPartitionedCall:output:0batch_normalization_4_388014batch_normalization_4_388016batch_normalization_4_388018batch_normalization_4_388020*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_386668ы
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_re_lu_4_layer_call_and_return_conditional_losses_387163ч
1spectral_normalization_14/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0 spectral_normalization_14_388024 spectral_normalization_14_388026 spectral_normalization_14_388028*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *^
fYRW
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_387462Є
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                Ѕ
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               Ѕ
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ї
(tf.__operators__.getitem_1/strided_sliceStridedSlice:spectral_normalization_14/StatefulPartitionedCall:output:07tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_maskў
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall1tf.__operators__.getitem_1/strided_slice:output:0batch_normalization_5_388035batch_normalization_5_388037batch_normalization_5_388039batch_normalization_5_388041*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_386776ы
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_re_lu_5_layer_call_and_return_conditional_losses_387212П
)self_attn_model_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0self_attn_model_1_388045self_attn_model_1_388047self_attn_model_1_388049self_attn_model_1_388051self_attn_model_1_388053self_attn_model_1_388055self_attn_model_1_388057*
Tin

2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:         :         DD*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_386988Ї
1spectral_normalization_15/StatefulPartitionedCallStatefulPartitionedCall2self_attn_model_1/StatefulPartitionedCall:output:0 spectral_normalization_15_388061 spectral_normalization_15_388063 spectral_normalization_15_388065*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *^
fYRW
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_387383Љ
IdentityIdentity:spectral_normalization_15/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         м
NoOpNoOp.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall*^self_attn_model_1/StatefulPartitionedCall2^spectral_normalization_12/StatefulPartitionedCall2^spectral_normalization_13/StatefulPartitionedCall2^spectral_normalization_14/StatefulPartitionedCall2^spectral_normalization_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:         ђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2V
)self_attn_model_1/StatefulPartitionedCall)self_attn_model_1/StatefulPartitionedCall2f
1spectral_normalization_12/StatefulPartitionedCall1spectral_normalization_12/StatefulPartitionedCall2f
1spectral_normalization_13/StatefulPartitionedCall1spectral_normalization_13/StatefulPartitionedCall2f
1spectral_normalization_14/StatefulPartitionedCall1spectral_normalization_14/StatefulPartitionedCall2f
1spectral_normalization_15/StatefulPartitionedCall1spectral_normalization_15/StatefulPartitionedCall:Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_4
Є╦
ё,
C__inference_model_1_layer_call_and_return_conditional_losses_388774

inputsT
9spectral_normalization_12_reshape_readvariableop_resource:	`ђ^
Kspectral_normalization_12_spectral_normalize_matmul_readvariableop_resource:	ђZ
Lspectral_normalization_12_conv2d_transpose_4_biasadd_readvariableop_resource:`;
-batch_normalization_3_readvariableop_resource:`=
/batch_normalization_3_readvariableop_1_resource:`L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:`N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:`S
9spectral_normalization_13_reshape_readvariableop_resource:0`]
Kspectral_normalization_13_spectral_normalize_matmul_readvariableop_resource:`Z
Lspectral_normalization_13_conv2d_transpose_5_biasadd_readvariableop_resource:0;
-batch_normalization_4_readvariableop_resource:0=
/batch_normalization_4_readvariableop_1_resource:0L
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:0N
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:0S
9spectral_normalization_14_reshape_readvariableop_resource:0]
Kspectral_normalization_14_spectral_normalize_matmul_readvariableop_resource:0Z
Lspectral_normalization_14_conv2d_transpose_6_biasadd_readvariableop_resource:;
-batch_normalization_5_readvariableop_resource:=
/batch_normalization_5_readvariableop_1_resource:L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:S
9self_attn_model_1_conv2d_9_conv2d_readvariableop_resource:H
:self_attn_model_1_conv2d_9_biasadd_readvariableop_resource:T
:self_attn_model_1_conv2d_10_conv2d_readvariableop_resource:I
;self_attn_model_1_conv2d_10_biasadd_readvariableop_resource:T
:self_attn_model_1_conv2d_11_conv2d_readvariableop_resource:I
;self_attn_model_1_conv2d_11_biasadd_readvariableop_resource:N
Dself_attn_model_1_private__attention_1_mul_3_readvariableop_resource: S
9spectral_normalization_15_reshape_readvariableop_resource:]
Kspectral_normalization_15_spectral_normalize_matmul_readvariableop_resource:Z
Lspectral_normalization_15_conv2d_transpose_7_biasadd_readvariableop_resource:
identityѕб$batch_normalization_3/AssignNewValueб&batch_normalization_3/AssignNewValue_1б5batch_normalization_3/FusedBatchNormV3/ReadVariableOpб7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_3/ReadVariableOpб&batch_normalization_3/ReadVariableOp_1б$batch_normalization_4/AssignNewValueб&batch_normalization_4/AssignNewValue_1б5batch_normalization_4/FusedBatchNormV3/ReadVariableOpб7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_4/ReadVariableOpб&batch_normalization_4/ReadVariableOp_1б$batch_normalization_5/AssignNewValueб&batch_normalization_5/AssignNewValue_1б5batch_normalization_5/FusedBatchNormV3/ReadVariableOpб7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_5/ReadVariableOpб&batch_normalization_5/ReadVariableOp_1б2self_attn_model_1/conv2d_10/BiasAdd/ReadVariableOpб1self_attn_model_1/conv2d_10/Conv2D/ReadVariableOpб2self_attn_model_1/conv2d_11/BiasAdd/ReadVariableOpб1self_attn_model_1/conv2d_11/Conv2D/ReadVariableOpб1self_attn_model_1/conv2d_9/BiasAdd/ReadVariableOpб0self_attn_model_1/conv2d_9/Conv2D/ReadVariableOpб;self_attn_model_1/private__attention_1/Mul_3/ReadVariableOpб0spectral_normalization_12/Reshape/ReadVariableOpбCspectral_normalization_12/conv2d_transpose_4/BiasAdd/ReadVariableOpбLspectral_normalization_12/conv2d_transpose_4/conv2d_transpose/ReadVariableOpб=spectral_normalization_12/spectral_normalize/AssignVariableOpб?spectral_normalization_12/spectral_normalize/AssignVariableOp_1бBspectral_normalization_12/spectral_normalize/MatMul/ReadVariableOpб;spectral_normalization_12/spectral_normalize/ReadVariableOpб0spectral_normalization_13/Reshape/ReadVariableOpбCspectral_normalization_13/conv2d_transpose_5/BiasAdd/ReadVariableOpбLspectral_normalization_13/conv2d_transpose_5/conv2d_transpose/ReadVariableOpб=spectral_normalization_13/spectral_normalize/AssignVariableOpб?spectral_normalization_13/spectral_normalize/AssignVariableOp_1бBspectral_normalization_13/spectral_normalize/MatMul/ReadVariableOpб;spectral_normalization_13/spectral_normalize/ReadVariableOpб0spectral_normalization_14/Reshape/ReadVariableOpбCspectral_normalization_14/conv2d_transpose_6/BiasAdd/ReadVariableOpбLspectral_normalization_14/conv2d_transpose_6/conv2d_transpose/ReadVariableOpб=spectral_normalization_14/spectral_normalize/AssignVariableOpб?spectral_normalization_14/spectral_normalize/AssignVariableOp_1бBspectral_normalization_14/spectral_normalize/MatMul/ReadVariableOpб;spectral_normalization_14/spectral_normalize/ReadVariableOpб0spectral_normalization_15/Reshape/ReadVariableOpбCspectral_normalization_15/conv2d_transpose_7/BiasAdd/ReadVariableOpбLspectral_normalization_15/conv2d_transpose_7/conv2d_transpose/ReadVariableOpб=spectral_normalization_15/spectral_normalize/AssignVariableOpб?spectral_normalization_15/spectral_normalize/AssignVariableOp_1бBspectral_normalization_15/spectral_normalize/MatMul/ReadVariableOpб;spectral_normalization_15/spectral_normalize/ReadVariableOpE
reshape_1/ShapeShapeinputs*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :\
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :ђ█
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ђ
reshape_1/ReshapeReshapeinputs reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:         ђ│
0spectral_normalization_12/Reshape/ReadVariableOpReadVariableOp9spectral_normalization_12_reshape_readvariableop_resource*'
_output_shapes
:	`ђ*
dtype0x
'spectral_normalization_12/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ђ   ├
!spectral_normalization_12/ReshapeReshape8spectral_normalization_12/Reshape/ReadVariableOp:value:00spectral_normalization_12/Reshape/shape:output:0*
T0* 
_output_shapes
:
ђђ¤
Bspectral_normalization_12/spectral_normalize/MatMul/ReadVariableOpReadVariableOpKspectral_normalization_12_spectral_normalize_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0Ы
3spectral_normalization_12/spectral_normalize/MatMulMatMulJspectral_normalization_12/spectral_normalize/MatMul/ReadVariableOp:value:0*spectral_normalization_12/Reshape:output:0*
T0*
_output_shapes
:	ђ*
transpose_b(│
@spectral_normalization_12/spectral_normalize/l2_normalize/SquareSquare=spectral_normalization_12/spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	ђљ
?spectral_normalization_12/spectral_normalize/l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ј
=spectral_normalization_12/spectral_normalize/l2_normalize/SumSumDspectral_normalization_12/spectral_normalize/l2_normalize/Square:y:0Hspectral_normalization_12/spectral_normalize/l2_normalize/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(ѕ
Cspectral_normalization_12/spectral_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+І
Aspectral_normalization_12/spectral_normalize/l2_normalize/MaximumMaximumFspectral_normalization_12/spectral_normalize/l2_normalize/Sum:output:0Lspectral_normalization_12/spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:И
?spectral_normalization_12/spectral_normalize/l2_normalize/RsqrtRsqrtEspectral_normalization_12/spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:Ь
9spectral_normalization_12/spectral_normalize/l2_normalizeMul=spectral_normalization_12/spectral_normalize/MatMul:product:0Cspectral_normalization_12/spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes
:	ђн
5spectral_normalization_12/spectral_normalize/MatMul_1MatMul=spectral_normalization_12/spectral_normalize/l2_normalize:z:0*spectral_normalization_12/Reshape:output:0*
T0*
_output_shapes
:	ђи
Bspectral_normalization_12/spectral_normalize/l2_normalize_1/SquareSquare?spectral_normalization_12/spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes
:	ђњ
Aspectral_normalization_12/spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ћ
?spectral_normalization_12/spectral_normalize/l2_normalize_1/SumSumFspectral_normalization_12/spectral_normalize/l2_normalize_1/Square:y:0Jspectral_normalization_12/spectral_normalize/l2_normalize_1/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(і
Espectral_normalization_12/spectral_normalize/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+Љ
Cspectral_normalization_12/spectral_normalize/l2_normalize_1/MaximumMaximumHspectral_normalization_12/spectral_normalize/l2_normalize_1/Sum:output:0Nspectral_normalization_12/spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:╝
Aspectral_normalization_12/spectral_normalize/l2_normalize_1/RsqrtRsqrtGspectral_normalization_12/spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:З
;spectral_normalization_12/spectral_normalize/l2_normalize_1Mul?spectral_normalization_12/spectral_normalize/MatMul_1:product:0Espectral_normalization_12/spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes
:	ђ┤
9spectral_normalization_12/spectral_normalize/StopGradientStopGradient?spectral_normalization_12/spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes
:	ђ┤
;spectral_normalization_12/spectral_normalize/StopGradient_1StopGradient=spectral_normalization_12/spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	ђ█
5spectral_normalization_12/spectral_normalize/MatMul_2MatMulDspectral_normalization_12/spectral_normalize/StopGradient_1:output:0*spectral_normalization_12/Reshape:output:0*
T0*
_output_shapes
:	ђђ
5spectral_normalization_12/spectral_normalize/MatMul_3MatMul?spectral_normalization_12/spectral_normalize/MatMul_2:product:0Bspectral_normalization_12/spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(Я
=spectral_normalization_12/spectral_normalize/AssignVariableOpAssignVariableOpKspectral_normalization_12_spectral_normalize_matmul_readvariableop_resourceBspectral_normalization_12/spectral_normalize/StopGradient:output:0C^spectral_normalization_12/spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Й
;spectral_normalization_12/spectral_normalize/ReadVariableOpReadVariableOp9spectral_normalization_12_reshape_readvariableop_resource*'
_output_shapes
:	`ђ*
dtype0э
4spectral_normalization_12/spectral_normalize/truedivRealDivCspectral_normalization_12/spectral_normalize/ReadVariableOp:value:0?spectral_normalization_12/spectral_normalize/MatMul_3:product:0*
T0*'
_output_shapes
:	`ђЊ
:spectral_normalization_12/spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   	   `   ђ   ­
4spectral_normalization_12/spectral_normalize/ReshapeReshape8spectral_normalization_12/spectral_normalize/truediv:z:0Cspectral_normalization_12/spectral_normalize/Reshape/shape:output:0*
T0*'
_output_shapes
:	`ђэ
?spectral_normalization_12/spectral_normalize/AssignVariableOp_1AssignVariableOp9spectral_normalization_12_reshape_readvariableop_resource=spectral_normalization_12/spectral_normalize/Reshape:output:01^spectral_normalization_12/Reshape/ReadVariableOp<^spectral_normalization_12/spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(|
2spectral_normalization_12/conv2d_transpose_4/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
:і
@spectral_normalization_12/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ї
Bspectral_normalization_12/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ї
Bspectral_normalization_12/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▓
:spectral_normalization_12/conv2d_transpose_4/strided_sliceStridedSlice;spectral_normalization_12/conv2d_transpose_4/Shape:output:0Ispectral_normalization_12/conv2d_transpose_4/strided_slice/stack:output:0Kspectral_normalization_12/conv2d_transpose_4/strided_slice/stack_1:output:0Kspectral_normalization_12/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4spectral_normalization_12/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :v
4spectral_normalization_12/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :v
4spectral_normalization_12/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :`Ж
2spectral_normalization_12/conv2d_transpose_4/stackPackCspectral_normalization_12/conv2d_transpose_4/strided_slice:output:0=spectral_normalization_12/conv2d_transpose_4/stack/1:output:0=spectral_normalization_12/conv2d_transpose_4/stack/2:output:0=spectral_normalization_12/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:ї
Bspectral_normalization_12/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ј
Dspectral_normalization_12/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ј
Dspectral_normalization_12/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
<spectral_normalization_12/conv2d_transpose_4/strided_slice_1StridedSlice;spectral_normalization_12/conv2d_transpose_4/stack:output:0Kspectral_normalization_12/conv2d_transpose_4/strided_slice_1/stack:output:0Mspectral_normalization_12/conv2d_transpose_4/strided_slice_1/stack_1:output:0Mspectral_normalization_12/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЉ
Lspectral_normalization_12/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp9spectral_normalization_12_reshape_readvariableop_resource@^spectral_normalization_12/spectral_normalize/AssignVariableOp_1*'
_output_shapes
:	`ђ*
dtype0т
=spectral_normalization_12/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput;spectral_normalization_12/conv2d_transpose_4/stack:output:0Tspectral_normalization_12/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0reshape_1/Reshape:output:0*
T0*/
_output_shapes
:         `*
paddingSAME*
strides
╠
Cspectral_normalization_12/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOpLspectral_normalization_12_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0ј
4spectral_normalization_12/conv2d_transpose_4/BiasAddBiasAddFspectral_normalization_12/conv2d_transpose_4/conv2d_transpose:output:0Kspectral_normalization_12/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `ј
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:`*
dtype0њ
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:`*
dtype0░
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0┤
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0ж
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3=spectral_normalization_12/conv2d_transpose_4/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         `:`:`:`:`:*
epsilon%oЃ:*
exponential_avg_factor%
О#<ъ
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(е
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(z
re_lu_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         `▓
0spectral_normalization_13/Reshape/ReadVariableOpReadVariableOp9spectral_normalization_13_reshape_readvariableop_resource*&
_output_shapes
:0`*
dtype0x
'spectral_normalization_13/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   ┬
!spectral_normalization_13/ReshapeReshape8spectral_normalization_13/Reshape/ReadVariableOp:value:00spectral_normalization_13/Reshape/shape:output:0*
T0*
_output_shapes
:	Я`╬
Bspectral_normalization_13/spectral_normalize/MatMul/ReadVariableOpReadVariableOpKspectral_normalization_13_spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:`*
dtype0Ы
3spectral_normalization_13/spectral_normalize/MatMulMatMulJspectral_normalization_13/spectral_normalize/MatMul/ReadVariableOp:value:0*spectral_normalization_13/Reshape:output:0*
T0*
_output_shapes
:	Я*
transpose_b(│
@spectral_normalization_13/spectral_normalize/l2_normalize/SquareSquare=spectral_normalization_13/spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	Яљ
?spectral_normalization_13/spectral_normalize/l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ј
=spectral_normalization_13/spectral_normalize/l2_normalize/SumSumDspectral_normalization_13/spectral_normalize/l2_normalize/Square:y:0Hspectral_normalization_13/spectral_normalize/l2_normalize/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(ѕ
Cspectral_normalization_13/spectral_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+І
Aspectral_normalization_13/spectral_normalize/l2_normalize/MaximumMaximumFspectral_normalization_13/spectral_normalize/l2_normalize/Sum:output:0Lspectral_normalization_13/spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:И
?spectral_normalization_13/spectral_normalize/l2_normalize/RsqrtRsqrtEspectral_normalization_13/spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:Ь
9spectral_normalization_13/spectral_normalize/l2_normalizeMul=spectral_normalization_13/spectral_normalize/MatMul:product:0Cspectral_normalization_13/spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes
:	ЯМ
5spectral_normalization_13/spectral_normalize/MatMul_1MatMul=spectral_normalization_13/spectral_normalize/l2_normalize:z:0*spectral_normalization_13/Reshape:output:0*
T0*
_output_shapes

:`Х
Bspectral_normalization_13/spectral_normalize/l2_normalize_1/SquareSquare?spectral_normalization_13/spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:`њ
Aspectral_normalization_13/spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ћ
?spectral_normalization_13/spectral_normalize/l2_normalize_1/SumSumFspectral_normalization_13/spectral_normalize/l2_normalize_1/Square:y:0Jspectral_normalization_13/spectral_normalize/l2_normalize_1/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(і
Espectral_normalization_13/spectral_normalize/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+Љ
Cspectral_normalization_13/spectral_normalize/l2_normalize_1/MaximumMaximumHspectral_normalization_13/spectral_normalize/l2_normalize_1/Sum:output:0Nspectral_normalization_13/spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:╝
Aspectral_normalization_13/spectral_normalize/l2_normalize_1/RsqrtRsqrtGspectral_normalization_13/spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:з
;spectral_normalization_13/spectral_normalize/l2_normalize_1Mul?spectral_normalization_13/spectral_normalize/MatMul_1:product:0Espectral_normalization_13/spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

:`│
9spectral_normalization_13/spectral_normalize/StopGradientStopGradient?spectral_normalization_13/spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:`┤
;spectral_normalization_13/spectral_normalize/StopGradient_1StopGradient=spectral_normalization_13/spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	Я┌
5spectral_normalization_13/spectral_normalize/MatMul_2MatMulDspectral_normalization_13/spectral_normalize/StopGradient_1:output:0*spectral_normalization_13/Reshape:output:0*
T0*
_output_shapes

:`ђ
5spectral_normalization_13/spectral_normalize/MatMul_3MatMul?spectral_normalization_13/spectral_normalize/MatMul_2:product:0Bspectral_normalization_13/spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(Я
=spectral_normalization_13/spectral_normalize/AssignVariableOpAssignVariableOpKspectral_normalization_13_spectral_normalize_matmul_readvariableop_resourceBspectral_normalization_13/spectral_normalize/StopGradient:output:0C^spectral_normalization_13/spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(й
;spectral_normalization_13/spectral_normalize/ReadVariableOpReadVariableOp9spectral_normalization_13_reshape_readvariableop_resource*&
_output_shapes
:0`*
dtype0Ш
4spectral_normalization_13/spectral_normalize/truedivRealDivCspectral_normalization_13/spectral_normalize/ReadVariableOp:value:0?spectral_normalization_13/spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:0`Њ
:spectral_normalization_13/spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      0   `   №
4spectral_normalization_13/spectral_normalize/ReshapeReshape8spectral_normalization_13/spectral_normalize/truediv:z:0Cspectral_normalization_13/spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:0`э
?spectral_normalization_13/spectral_normalize/AssignVariableOp_1AssignVariableOp9spectral_normalization_13_reshape_readvariableop_resource=spectral_normalization_13/spectral_normalize/Reshape:output:01^spectral_normalization_13/Reshape/ReadVariableOp<^spectral_normalization_13/spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(|
2spectral_normalization_13/conv2d_transpose_5/ShapeShapere_lu_3/Relu:activations:0*
T0*
_output_shapes
:і
@spectral_normalization_13/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ї
Bspectral_normalization_13/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ї
Bspectral_normalization_13/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▓
:spectral_normalization_13/conv2d_transpose_5/strided_sliceStridedSlice;spectral_normalization_13/conv2d_transpose_5/Shape:output:0Ispectral_normalization_13/conv2d_transpose_5/strided_slice/stack:output:0Kspectral_normalization_13/conv2d_transpose_5/strided_slice/stack_1:output:0Kspectral_normalization_13/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4spectral_normalization_13/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :v
4spectral_normalization_13/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :	v
4spectral_normalization_13/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :0Ж
2spectral_normalization_13/conv2d_transpose_5/stackPackCspectral_normalization_13/conv2d_transpose_5/strided_slice:output:0=spectral_normalization_13/conv2d_transpose_5/stack/1:output:0=spectral_normalization_13/conv2d_transpose_5/stack/2:output:0=spectral_normalization_13/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:ї
Bspectral_normalization_13/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ј
Dspectral_normalization_13/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ј
Dspectral_normalization_13/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
<spectral_normalization_13/conv2d_transpose_5/strided_slice_1StridedSlice;spectral_normalization_13/conv2d_transpose_5/stack:output:0Kspectral_normalization_13/conv2d_transpose_5/strided_slice_1/stack:output:0Mspectral_normalization_13/conv2d_transpose_5/strided_slice_1/stack_1:output:0Mspectral_normalization_13/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskљ
Lspectral_normalization_13/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp9spectral_normalization_13_reshape_readvariableop_resource@^spectral_normalization_13/spectral_normalize/AssignVariableOp_1*&
_output_shapes
:0`*
dtype0т
=spectral_normalization_13/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput;spectral_normalization_13/conv2d_transpose_5/stack:output:0Tspectral_normalization_13/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0re_lu_3/Relu:activations:0*
T0*/
_output_shapes
:         	0*
paddingSAME*
strides
╠
Cspectral_normalization_13/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOpLspectral_normalization_13_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0ј
4spectral_normalization_13/conv2d_transpose_5/BiasAddBiasAddFspectral_normalization_13/conv2d_transpose_5/conv2d_transpose:output:0Kspectral_normalization_13/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         	0ј
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:0*
dtype0њ
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:0*
dtype0░
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0┤
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0ж
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3=spectral_normalization_13/conv2d_transpose_5/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         	0:0:0:0:0:*
epsilon%oЃ:*
exponential_avg_factor%
О#<ъ
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(е
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(z
re_lu_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         	0▓
0spectral_normalization_14/Reshape/ReadVariableOpReadVariableOp9spectral_normalization_14_reshape_readvariableop_resource*&
_output_shapes
:0*
dtype0x
'spectral_normalization_14/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    0   ┴
!spectral_normalization_14/ReshapeReshape8spectral_normalization_14/Reshape/ReadVariableOp:value:00spectral_normalization_14/Reshape/shape:output:0*
T0*
_output_shapes

:H0╬
Bspectral_normalization_14/spectral_normalize/MatMul/ReadVariableOpReadVariableOpKspectral_normalization_14_spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:0*
dtype0ы
3spectral_normalization_14/spectral_normalize/MatMulMatMulJspectral_normalization_14/spectral_normalize/MatMul/ReadVariableOp:value:0*spectral_normalization_14/Reshape:output:0*
T0*
_output_shapes

:H*
transpose_b(▓
@spectral_normalization_14/spectral_normalize/l2_normalize/SquareSquare=spectral_normalization_14/spectral_normalize/MatMul:product:0*
T0*
_output_shapes

:Hљ
?spectral_normalization_14/spectral_normalize/l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ј
=spectral_normalization_14/spectral_normalize/l2_normalize/SumSumDspectral_normalization_14/spectral_normalize/l2_normalize/Square:y:0Hspectral_normalization_14/spectral_normalize/l2_normalize/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(ѕ
Cspectral_normalization_14/spectral_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+І
Aspectral_normalization_14/spectral_normalize/l2_normalize/MaximumMaximumFspectral_normalization_14/spectral_normalize/l2_normalize/Sum:output:0Lspectral_normalization_14/spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:И
?spectral_normalization_14/spectral_normalize/l2_normalize/RsqrtRsqrtEspectral_normalization_14/spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:ь
9spectral_normalization_14/spectral_normalize/l2_normalizeMul=spectral_normalization_14/spectral_normalize/MatMul:product:0Cspectral_normalization_14/spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes

:HМ
5spectral_normalization_14/spectral_normalize/MatMul_1MatMul=spectral_normalization_14/spectral_normalize/l2_normalize:z:0*spectral_normalization_14/Reshape:output:0*
T0*
_output_shapes

:0Х
Bspectral_normalization_14/spectral_normalize/l2_normalize_1/SquareSquare?spectral_normalization_14/spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:0њ
Aspectral_normalization_14/spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ћ
?spectral_normalization_14/spectral_normalize/l2_normalize_1/SumSumFspectral_normalization_14/spectral_normalize/l2_normalize_1/Square:y:0Jspectral_normalization_14/spectral_normalize/l2_normalize_1/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(і
Espectral_normalization_14/spectral_normalize/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+Љ
Cspectral_normalization_14/spectral_normalize/l2_normalize_1/MaximumMaximumHspectral_normalization_14/spectral_normalize/l2_normalize_1/Sum:output:0Nspectral_normalization_14/spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:╝
Aspectral_normalization_14/spectral_normalize/l2_normalize_1/RsqrtRsqrtGspectral_normalization_14/spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:з
;spectral_normalization_14/spectral_normalize/l2_normalize_1Mul?spectral_normalization_14/spectral_normalize/MatMul_1:product:0Espectral_normalization_14/spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

:0│
9spectral_normalization_14/spectral_normalize/StopGradientStopGradient?spectral_normalization_14/spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:0│
;spectral_normalization_14/spectral_normalize/StopGradient_1StopGradient=spectral_normalization_14/spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes

:H┌
5spectral_normalization_14/spectral_normalize/MatMul_2MatMulDspectral_normalization_14/spectral_normalize/StopGradient_1:output:0*spectral_normalization_14/Reshape:output:0*
T0*
_output_shapes

:0ђ
5spectral_normalization_14/spectral_normalize/MatMul_3MatMul?spectral_normalization_14/spectral_normalize/MatMul_2:product:0Bspectral_normalization_14/spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(Я
=spectral_normalization_14/spectral_normalize/AssignVariableOpAssignVariableOpKspectral_normalization_14_spectral_normalize_matmul_readvariableop_resourceBspectral_normalization_14/spectral_normalize/StopGradient:output:0C^spectral_normalization_14/spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(й
;spectral_normalization_14/spectral_normalize/ReadVariableOpReadVariableOp9spectral_normalization_14_reshape_readvariableop_resource*&
_output_shapes
:0*
dtype0Ш
4spectral_normalization_14/spectral_normalize/truedivRealDivCspectral_normalization_14/spectral_normalize/ReadVariableOp:value:0?spectral_normalization_14/spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:0Њ
:spectral_normalization_14/spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         0   №
4spectral_normalization_14/spectral_normalize/ReshapeReshape8spectral_normalization_14/spectral_normalize/truediv:z:0Cspectral_normalization_14/spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:0э
?spectral_normalization_14/spectral_normalize/AssignVariableOp_1AssignVariableOp9spectral_normalization_14_reshape_readvariableop_resource=spectral_normalization_14/spectral_normalize/Reshape:output:01^spectral_normalization_14/Reshape/ReadVariableOp<^spectral_normalization_14/spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(|
2spectral_normalization_14/conv2d_transpose_6/ShapeShapere_lu_4/Relu:activations:0*
T0*
_output_shapes
:і
@spectral_normalization_14/conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ї
Bspectral_normalization_14/conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ї
Bspectral_normalization_14/conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▓
:spectral_normalization_14/conv2d_transpose_6/strided_sliceStridedSlice;spectral_normalization_14/conv2d_transpose_6/Shape:output:0Ispectral_normalization_14/conv2d_transpose_6/strided_slice/stack:output:0Kspectral_normalization_14/conv2d_transpose_6/strided_slice/stack_1:output:0Kspectral_normalization_14/conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4spectral_normalization_14/conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :v
4spectral_normalization_14/conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :v
4spectral_normalization_14/conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :Ж
2spectral_normalization_14/conv2d_transpose_6/stackPackCspectral_normalization_14/conv2d_transpose_6/strided_slice:output:0=spectral_normalization_14/conv2d_transpose_6/stack/1:output:0=spectral_normalization_14/conv2d_transpose_6/stack/2:output:0=spectral_normalization_14/conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:ї
Bspectral_normalization_14/conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ј
Dspectral_normalization_14/conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ј
Dspectral_normalization_14/conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
<spectral_normalization_14/conv2d_transpose_6/strided_slice_1StridedSlice;spectral_normalization_14/conv2d_transpose_6/stack:output:0Kspectral_normalization_14/conv2d_transpose_6/strided_slice_1/stack:output:0Mspectral_normalization_14/conv2d_transpose_6/strided_slice_1/stack_1:output:0Mspectral_normalization_14/conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskљ
Lspectral_normalization_14/conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOp9spectral_normalization_14_reshape_readvariableop_resource@^spectral_normalization_14/spectral_normalize/AssignVariableOp_1*&
_output_shapes
:0*
dtype0т
=spectral_normalization_14/conv2d_transpose_6/conv2d_transposeConv2DBackpropInput;spectral_normalization_14/conv2d_transpose_6/stack:output:0Tspectral_normalization_14/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0re_lu_4/Relu:activations:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
╠
Cspectral_normalization_14/conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOpLspectral_normalization_14_conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ј
4spectral_normalization_14/conv2d_transpose_6/BiasAddBiasAddFspectral_normalization_14/conv2d_transpose_6/conv2d_transpose:output:0Kspectral_normalization_14/conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         Є
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                Ѕ
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               Ѕ
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ј
(tf.__operators__.getitem_1/strided_sliceStridedSlice=spectral_normalization_14/conv2d_transpose_6/BiasAdd:output:07tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_maskј
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype0њ
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype0░
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0┤
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0П
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV31tf.__operators__.getitem_1/strided_slice:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<ъ
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(е
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(z
re_lu_5/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         ▓
0self_attn_model_1/conv2d_9/Conv2D/ReadVariableOpReadVariableOp9self_attn_model_1_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0С
!self_attn_model_1/conv2d_9/Conv2DConv2Dre_lu_5/Relu:activations:08self_attn_model_1/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
е
1self_attn_model_1/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp:self_attn_model_1_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╬
"self_attn_model_1/conv2d_9/BiasAddBiasAdd*self_attn_model_1/conv2d_9/Conv2D:output:09self_attn_model_1/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ┤
1self_attn_model_1/conv2d_10/Conv2D/ReadVariableOpReadVariableOp:self_attn_model_1_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Т
"self_attn_model_1/conv2d_10/Conv2DConv2Dre_lu_5/Relu:activations:09self_attn_model_1/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
ф
2self_attn_model_1/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp;self_attn_model_1_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
#self_attn_model_1/conv2d_10/BiasAddBiasAdd+self_attn_model_1/conv2d_10/Conv2D:output:0:self_attn_model_1/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ┤
1self_attn_model_1/conv2d_11/Conv2D/ReadVariableOpReadVariableOp:self_attn_model_1_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Т
"self_attn_model_1/conv2d_11/Conv2DConv2Dre_lu_5/Relu:activations:09self_attn_model_1/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
ф
2self_attn_model_1/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp;self_attn_model_1_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
#self_attn_model_1/conv2d_11/BiasAddBiasAdd+self_attn_model_1/conv2d_11/Conv2D:output:0:self_attn_model_1/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         Є
,self_attn_model_1/private__attention_1/ShapeShape+self_attn_model_1/conv2d_9/BiasAdd:output:0*
T0*
_output_shapes
:ё
:self_attn_model_1/private__attention_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: є
<self_attn_model_1/private__attention_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:є
<self_attn_model_1/private__attention_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ћ
4self_attn_model_1/private__attention_1/strided_sliceStridedSlice5self_attn_model_1/private__attention_1/Shape:output:0Cself_attn_model_1/private__attention_1/strided_slice/stack:output:0Eself_attn_model_1/private__attention_1/strided_slice/stack_1:output:0Eself_attn_model_1/private__attention_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskє
<self_attn_model_1/private__attention_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:ѕ
>self_attn_model_1/private__attention_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ѕ
>self_attn_model_1/private__attention_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
6self_attn_model_1/private__attention_1/strided_slice_1StridedSlice5self_attn_model_1/private__attention_1/Shape:output:0Eself_attn_model_1/private__attention_1/strided_slice_1/stack:output:0Gself_attn_model_1/private__attention_1/strided_slice_1/stack_1:output:0Gself_attn_model_1/private__attention_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskє
<self_attn_model_1/private__attention_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:ѕ
>self_attn_model_1/private__attention_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ѕ
>self_attn_model_1/private__attention_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
6self_attn_model_1/private__attention_1/strided_slice_2StridedSlice5self_attn_model_1/private__attention_1/Shape:output:0Eself_attn_model_1/private__attention_1/strided_slice_2/stack:output:0Gself_attn_model_1/private__attention_1/strided_slice_2/stack_1:output:0Gself_attn_model_1/private__attention_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskн
*self_attn_model_1/private__attention_1/mulMul?self_attn_model_1/private__attention_1/strided_slice_1:output:0?self_attn_model_1/private__attention_1/strided_slice_2:output:0*
T0*
_output_shapes
: Ђ
6self_attn_model_1/private__attention_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
         џ
4self_attn_model_1/private__attention_1/Reshape/shapePack=self_attn_model_1/private__attention_1/strided_slice:output:0.self_attn_model_1/private__attention_1/mul:z:0?self_attn_model_1/private__attention_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:С
.self_attn_model_1/private__attention_1/ReshapeReshape+self_attn_model_1/conv2d_9/BiasAdd:output:0=self_attn_model_1/private__attention_1/Reshape/shape:output:0*
T0*4
_output_shapes"
 :         D         о
,self_attn_model_1/private__attention_1/mul_1Mul?self_attn_model_1/private__attention_1/strided_slice_1:output:0?self_attn_model_1/private__attention_1/strided_slice_2:output:0*
T0*
_output_shapes
: Ѓ
8self_attn_model_1/private__attention_1/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
         а
6self_attn_model_1/private__attention_1/Reshape_1/shapePack=self_attn_model_1/private__attention_1/strided_slice:output:00self_attn_model_1/private__attention_1/mul_1:z:0Aself_attn_model_1/private__attention_1/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:ж
0self_attn_model_1/private__attention_1/Reshape_1Reshape,self_attn_model_1/conv2d_10/BiasAdd:output:0?self_attn_model_1/private__attention_1/Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :         D         і
5self_attn_model_1/private__attention_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          э
0self_attn_model_1/private__attention_1/transpose	Transpose9self_attn_model_1/private__attention_1/Reshape_1:output:0>self_attn_model_1/private__attention_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :                  Dо
,self_attn_model_1/private__attention_1/mul_2Mul?self_attn_model_1/private__attention_1/strided_slice_1:output:0?self_attn_model_1/private__attention_1/strided_slice_2:output:0*
T0*
_output_shapes
: Ѓ
8self_attn_model_1/private__attention_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
         а
6self_attn_model_1/private__attention_1/Reshape_2/shapePack=self_attn_model_1/private__attention_1/strided_slice:output:00self_attn_model_1/private__attention_1/mul_2:z:0Aself_attn_model_1/private__attention_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:ж
0self_attn_model_1/private__attention_1/Reshape_2Reshape,self_attn_model_1/conv2d_11/BiasAdd:output:0?self_attn_model_1/private__attention_1/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :         D         ї
7self_attn_model_1/private__attention_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ч
2self_attn_model_1/private__attention_1/transpose_1	Transpose9self_attn_model_1/private__attention_1/Reshape_2:output:0@self_attn_model_1/private__attention_1/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  Dс
-self_attn_model_1/private__attention_1/MatMulBatchMatMulV27self_attn_model_1/private__attention_1/Reshape:output:04self_attn_model_1/private__attention_1/transpose:y:0*
T0*+
_output_shapes
:         DDД
.self_attn_model_1/private__attention_1/SoftmaxSoftmax6self_attn_model_1/private__attention_1/MatMul:output:0*
T0*+
_output_shapes
:         DDї
7self_attn_model_1/private__attention_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ы
2self_attn_model_1/private__attention_1/transpose_2	Transpose8self_attn_model_1/private__attention_1/Softmax:softmax:0@self_attn_model_1/private__attention_1/transpose_2/perm:output:0*
T0*+
_output_shapes
:         DD№
/self_attn_model_1/private__attention_1/MatMul_1BatchMatMulV26self_attn_model_1/private__attention_1/transpose_1:y:06self_attn_model_1/private__attention_1/transpose_2:y:0*
T0*4
_output_shapes"
 :                  Dї
7self_attn_model_1/private__attention_1/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          Щ
2self_attn_model_1/private__attention_1/transpose_3	Transpose8self_attn_model_1/private__attention_1/MatMul_1:output:0@self_attn_model_1/private__attention_1/transpose_3/perm:output:0*
T0*4
_output_shapes"
 :         D         Ѓ
8self_attn_model_1/private__attention_1/Reshape_3/shape/3Const*
_output_shapes
: *
dtype0*
valueB :
         ­
6self_attn_model_1/private__attention_1/Reshape_3/shapePack=self_attn_model_1/private__attention_1/strided_slice:output:0?self_attn_model_1/private__attention_1/strided_slice_1:output:0?self_attn_model_1/private__attention_1/strided_slice_2:output:0Aself_attn_model_1/private__attention_1/Reshape_3/shape/3:output:0*
N*
T0*
_output_shapes
:э
0self_attn_model_1/private__attention_1/Reshape_3Reshape6self_attn_model_1/private__attention_1/transpose_3:y:0?self_attn_model_1/private__attention_1/Reshape_3/shape:output:0*
T0*8
_output_shapes&
$:"                  И
;self_attn_model_1/private__attention_1/Mul_3/ReadVariableOpReadVariableOpDself_attn_model_1_private__attention_1_mul_3_readvariableop_resource*
_output_shapes
: *
dtype0Ш
,self_attn_model_1/private__attention_1/Mul_3Mul9self_attn_model_1/private__attention_1/Reshape_3:output:0Cself_attn_model_1/private__attention_1/Mul_3/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  ╗
*self_attn_model_1/private__attention_1/AddAddV20self_attn_model_1/private__attention_1/Mul_3:z:0re_lu_5/Relu:activations:0*
T0*/
_output_shapes
:         ▓
0spectral_normalization_15/Reshape/ReadVariableOpReadVariableOp9spectral_normalization_15_reshape_readvariableop_resource*&
_output_shapes
:*
dtype0x
'spectral_normalization_15/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┴
!spectral_normalization_15/ReshapeReshape8spectral_normalization_15/Reshape/ReadVariableOp:value:00spectral_normalization_15/Reshape/shape:output:0*
T0*
_output_shapes

:╬
Bspectral_normalization_15/spectral_normalize/MatMul/ReadVariableOpReadVariableOpKspectral_normalization_15_spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0ы
3spectral_normalization_15/spectral_normalize/MatMulMatMulJspectral_normalization_15/spectral_normalize/MatMul/ReadVariableOp:value:0*spectral_normalization_15/Reshape:output:0*
T0*
_output_shapes

:*
transpose_b(▓
@spectral_normalization_15/spectral_normalize/l2_normalize/SquareSquare=spectral_normalization_15/spectral_normalize/MatMul:product:0*
T0*
_output_shapes

:љ
?spectral_normalization_15/spectral_normalize/l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ј
=spectral_normalization_15/spectral_normalize/l2_normalize/SumSumDspectral_normalization_15/spectral_normalize/l2_normalize/Square:y:0Hspectral_normalization_15/spectral_normalize/l2_normalize/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(ѕ
Cspectral_normalization_15/spectral_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+І
Aspectral_normalization_15/spectral_normalize/l2_normalize/MaximumMaximumFspectral_normalization_15/spectral_normalize/l2_normalize/Sum:output:0Lspectral_normalization_15/spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:И
?spectral_normalization_15/spectral_normalize/l2_normalize/RsqrtRsqrtEspectral_normalization_15/spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:ь
9spectral_normalization_15/spectral_normalize/l2_normalizeMul=spectral_normalization_15/spectral_normalize/MatMul:product:0Cspectral_normalization_15/spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes

:М
5spectral_normalization_15/spectral_normalize/MatMul_1MatMul=spectral_normalization_15/spectral_normalize/l2_normalize:z:0*spectral_normalization_15/Reshape:output:0*
T0*
_output_shapes

:Х
Bspectral_normalization_15/spectral_normalize/l2_normalize_1/SquareSquare?spectral_normalization_15/spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:њ
Aspectral_normalization_15/spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ћ
?spectral_normalization_15/spectral_normalize/l2_normalize_1/SumSumFspectral_normalization_15/spectral_normalize/l2_normalize_1/Square:y:0Jspectral_normalization_15/spectral_normalize/l2_normalize_1/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(і
Espectral_normalization_15/spectral_normalize/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+Љ
Cspectral_normalization_15/spectral_normalize/l2_normalize_1/MaximumMaximumHspectral_normalization_15/spectral_normalize/l2_normalize_1/Sum:output:0Nspectral_normalization_15/spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:╝
Aspectral_normalization_15/spectral_normalize/l2_normalize_1/RsqrtRsqrtGspectral_normalization_15/spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:з
;spectral_normalization_15/spectral_normalize/l2_normalize_1Mul?spectral_normalization_15/spectral_normalize/MatMul_1:product:0Espectral_normalization_15/spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

:│
9spectral_normalization_15/spectral_normalize/StopGradientStopGradient?spectral_normalization_15/spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:│
;spectral_normalization_15/spectral_normalize/StopGradient_1StopGradient=spectral_normalization_15/spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes

:┌
5spectral_normalization_15/spectral_normalize/MatMul_2MatMulDspectral_normalization_15/spectral_normalize/StopGradient_1:output:0*spectral_normalization_15/Reshape:output:0*
T0*
_output_shapes

:ђ
5spectral_normalization_15/spectral_normalize/MatMul_3MatMul?spectral_normalization_15/spectral_normalize/MatMul_2:product:0Bspectral_normalization_15/spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(Я
=spectral_normalization_15/spectral_normalize/AssignVariableOpAssignVariableOpKspectral_normalization_15_spectral_normalize_matmul_readvariableop_resourceBspectral_normalization_15/spectral_normalize/StopGradient:output:0C^spectral_normalization_15/spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(й
;spectral_normalization_15/spectral_normalize/ReadVariableOpReadVariableOp9spectral_normalization_15_reshape_readvariableop_resource*&
_output_shapes
:*
dtype0Ш
4spectral_normalization_15/spectral_normalize/truedivRealDivCspectral_normalization_15/spectral_normalize/ReadVariableOp:value:0?spectral_normalization_15/spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:Њ
:spectral_normalization_15/spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            №
4spectral_normalization_15/spectral_normalize/ReshapeReshape8spectral_normalization_15/spectral_normalize/truediv:z:0Cspectral_normalization_15/spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:э
?spectral_normalization_15/spectral_normalize/AssignVariableOp_1AssignVariableOp9spectral_normalization_15_reshape_readvariableop_resource=spectral_normalization_15/spectral_normalize/Reshape:output:01^spectral_normalization_15/Reshape/ReadVariableOp<^spectral_normalization_15/spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(љ
2spectral_normalization_15/conv2d_transpose_7/ShapeShape.self_attn_model_1/private__attention_1/Add:z:0*
T0*
_output_shapes
:і
@spectral_normalization_15/conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ї
Bspectral_normalization_15/conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ї
Bspectral_normalization_15/conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▓
:spectral_normalization_15/conv2d_transpose_7/strided_sliceStridedSlice;spectral_normalization_15/conv2d_transpose_7/Shape:output:0Ispectral_normalization_15/conv2d_transpose_7/strided_slice/stack:output:0Kspectral_normalization_15/conv2d_transpose_7/strided_slice/stack_1:output:0Kspectral_normalization_15/conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4spectral_normalization_15/conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :v
4spectral_normalization_15/conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :v
4spectral_normalization_15/conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :Ж
2spectral_normalization_15/conv2d_transpose_7/stackPackCspectral_normalization_15/conv2d_transpose_7/strided_slice:output:0=spectral_normalization_15/conv2d_transpose_7/stack/1:output:0=spectral_normalization_15/conv2d_transpose_7/stack/2:output:0=spectral_normalization_15/conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:ї
Bspectral_normalization_15/conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ј
Dspectral_normalization_15/conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ј
Dspectral_normalization_15/conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
<spectral_normalization_15/conv2d_transpose_7/strided_slice_1StridedSlice;spectral_normalization_15/conv2d_transpose_7/stack:output:0Kspectral_normalization_15/conv2d_transpose_7/strided_slice_1/stack:output:0Mspectral_normalization_15/conv2d_transpose_7/strided_slice_1/stack_1:output:0Mspectral_normalization_15/conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskљ
Lspectral_normalization_15/conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOp9spectral_normalization_15_reshape_readvariableop_resource@^spectral_normalization_15/spectral_normalize/AssignVariableOp_1*&
_output_shapes
:*
dtype0щ
=spectral_normalization_15/conv2d_transpose_7/conv2d_transposeConv2DBackpropInput;spectral_normalization_15/conv2d_transpose_7/stack:output:0Tspectral_normalization_15/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0.self_attn_model_1/private__attention_1/Add:z:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
╠
Cspectral_normalization_15/conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOpLspectral_normalization_15_conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ј
4spectral_normalization_15/conv2d_transpose_7/BiasAddBiasAddFspectral_normalization_15/conv2d_transpose_7/conv2d_transpose:output:0Kspectral_normalization_15/conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ▓
1spectral_normalization_15/conv2d_transpose_7/TanhTanh=spectral_normalization_15/conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:         ї
IdentityIdentity5spectral_normalization_15/conv2d_transpose_7/Tanh:y:0^NoOp*
T0*/
_output_shapes
:         Д
NoOpNoOp%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_13^self_attn_model_1/conv2d_10/BiasAdd/ReadVariableOp2^self_attn_model_1/conv2d_10/Conv2D/ReadVariableOp3^self_attn_model_1/conv2d_11/BiasAdd/ReadVariableOp2^self_attn_model_1/conv2d_11/Conv2D/ReadVariableOp2^self_attn_model_1/conv2d_9/BiasAdd/ReadVariableOp1^self_attn_model_1/conv2d_9/Conv2D/ReadVariableOp<^self_attn_model_1/private__attention_1/Mul_3/ReadVariableOp1^spectral_normalization_12/Reshape/ReadVariableOpD^spectral_normalization_12/conv2d_transpose_4/BiasAdd/ReadVariableOpM^spectral_normalization_12/conv2d_transpose_4/conv2d_transpose/ReadVariableOp>^spectral_normalization_12/spectral_normalize/AssignVariableOp@^spectral_normalization_12/spectral_normalize/AssignVariableOp_1C^spectral_normalization_12/spectral_normalize/MatMul/ReadVariableOp<^spectral_normalization_12/spectral_normalize/ReadVariableOp1^spectral_normalization_13/Reshape/ReadVariableOpD^spectral_normalization_13/conv2d_transpose_5/BiasAdd/ReadVariableOpM^spectral_normalization_13/conv2d_transpose_5/conv2d_transpose/ReadVariableOp>^spectral_normalization_13/spectral_normalize/AssignVariableOp@^spectral_normalization_13/spectral_normalize/AssignVariableOp_1C^spectral_normalization_13/spectral_normalize/MatMul/ReadVariableOp<^spectral_normalization_13/spectral_normalize/ReadVariableOp1^spectral_normalization_14/Reshape/ReadVariableOpD^spectral_normalization_14/conv2d_transpose_6/BiasAdd/ReadVariableOpM^spectral_normalization_14/conv2d_transpose_6/conv2d_transpose/ReadVariableOp>^spectral_normalization_14/spectral_normalize/AssignVariableOp@^spectral_normalization_14/spectral_normalize/AssignVariableOp_1C^spectral_normalization_14/spectral_normalize/MatMul/ReadVariableOp<^spectral_normalization_14/spectral_normalize/ReadVariableOp1^spectral_normalization_15/Reshape/ReadVariableOpD^spectral_normalization_15/conv2d_transpose_7/BiasAdd/ReadVariableOpM^spectral_normalization_15/conv2d_transpose_7/conv2d_transpose/ReadVariableOp>^spectral_normalization_15/spectral_normalize/AssignVariableOp@^spectral_normalization_15/spectral_normalize/AssignVariableOp_1C^spectral_normalization_15/spectral_normalize/MatMul/ReadVariableOp<^spectral_normalization_15/spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:         ђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12h
2self_attn_model_1/conv2d_10/BiasAdd/ReadVariableOp2self_attn_model_1/conv2d_10/BiasAdd/ReadVariableOp2f
1self_attn_model_1/conv2d_10/Conv2D/ReadVariableOp1self_attn_model_1/conv2d_10/Conv2D/ReadVariableOp2h
2self_attn_model_1/conv2d_11/BiasAdd/ReadVariableOp2self_attn_model_1/conv2d_11/BiasAdd/ReadVariableOp2f
1self_attn_model_1/conv2d_11/Conv2D/ReadVariableOp1self_attn_model_1/conv2d_11/Conv2D/ReadVariableOp2f
1self_attn_model_1/conv2d_9/BiasAdd/ReadVariableOp1self_attn_model_1/conv2d_9/BiasAdd/ReadVariableOp2d
0self_attn_model_1/conv2d_9/Conv2D/ReadVariableOp0self_attn_model_1/conv2d_9/Conv2D/ReadVariableOp2z
;self_attn_model_1/private__attention_1/Mul_3/ReadVariableOp;self_attn_model_1/private__attention_1/Mul_3/ReadVariableOp2d
0spectral_normalization_12/Reshape/ReadVariableOp0spectral_normalization_12/Reshape/ReadVariableOp2і
Cspectral_normalization_12/conv2d_transpose_4/BiasAdd/ReadVariableOpCspectral_normalization_12/conv2d_transpose_4/BiasAdd/ReadVariableOp2ю
Lspectral_normalization_12/conv2d_transpose_4/conv2d_transpose/ReadVariableOpLspectral_normalization_12/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2~
=spectral_normalization_12/spectral_normalize/AssignVariableOp=spectral_normalization_12/spectral_normalize/AssignVariableOp2ѓ
?spectral_normalization_12/spectral_normalize/AssignVariableOp_1?spectral_normalization_12/spectral_normalize/AssignVariableOp_12ѕ
Bspectral_normalization_12/spectral_normalize/MatMul/ReadVariableOpBspectral_normalization_12/spectral_normalize/MatMul/ReadVariableOp2z
;spectral_normalization_12/spectral_normalize/ReadVariableOp;spectral_normalization_12/spectral_normalize/ReadVariableOp2d
0spectral_normalization_13/Reshape/ReadVariableOp0spectral_normalization_13/Reshape/ReadVariableOp2і
Cspectral_normalization_13/conv2d_transpose_5/BiasAdd/ReadVariableOpCspectral_normalization_13/conv2d_transpose_5/BiasAdd/ReadVariableOp2ю
Lspectral_normalization_13/conv2d_transpose_5/conv2d_transpose/ReadVariableOpLspectral_normalization_13/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2~
=spectral_normalization_13/spectral_normalize/AssignVariableOp=spectral_normalization_13/spectral_normalize/AssignVariableOp2ѓ
?spectral_normalization_13/spectral_normalize/AssignVariableOp_1?spectral_normalization_13/spectral_normalize/AssignVariableOp_12ѕ
Bspectral_normalization_13/spectral_normalize/MatMul/ReadVariableOpBspectral_normalization_13/spectral_normalize/MatMul/ReadVariableOp2z
;spectral_normalization_13/spectral_normalize/ReadVariableOp;spectral_normalization_13/spectral_normalize/ReadVariableOp2d
0spectral_normalization_14/Reshape/ReadVariableOp0spectral_normalization_14/Reshape/ReadVariableOp2і
Cspectral_normalization_14/conv2d_transpose_6/BiasAdd/ReadVariableOpCspectral_normalization_14/conv2d_transpose_6/BiasAdd/ReadVariableOp2ю
Lspectral_normalization_14/conv2d_transpose_6/conv2d_transpose/ReadVariableOpLspectral_normalization_14/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2~
=spectral_normalization_14/spectral_normalize/AssignVariableOp=spectral_normalization_14/spectral_normalize/AssignVariableOp2ѓ
?spectral_normalization_14/spectral_normalize/AssignVariableOp_1?spectral_normalization_14/spectral_normalize/AssignVariableOp_12ѕ
Bspectral_normalization_14/spectral_normalize/MatMul/ReadVariableOpBspectral_normalization_14/spectral_normalize/MatMul/ReadVariableOp2z
;spectral_normalization_14/spectral_normalize/ReadVariableOp;spectral_normalization_14/spectral_normalize/ReadVariableOp2d
0spectral_normalization_15/Reshape/ReadVariableOp0spectral_normalization_15/Reshape/ReadVariableOp2і
Cspectral_normalization_15/conv2d_transpose_7/BiasAdd/ReadVariableOpCspectral_normalization_15/conv2d_transpose_7/BiasAdd/ReadVariableOp2ю
Lspectral_normalization_15/conv2d_transpose_7/conv2d_transpose/ReadVariableOpLspectral_normalization_15/conv2d_transpose_7/conv2d_transpose/ReadVariableOp2~
=spectral_normalization_15/spectral_normalize/AssignVariableOp=spectral_normalization_15/spectral_normalize/AssignVariableOp2ѓ
?spectral_normalization_15/spectral_normalize/AssignVariableOp_1?spectral_normalization_15/spectral_normalize/AssignVariableOp_12ѕ
Bspectral_normalization_15/spectral_normalize/MatMul/ReadVariableOpBspectral_normalization_15/spectral_normalize/MatMul/ReadVariableOp2z
;spectral_normalization_15/spectral_normalize/ReadVariableOp;spectral_normalization_15/spectral_normalize/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
┴
Ь
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_387254

inputsU
;conv2d_transpose_7_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_7_biasadd_readvariableop_resource:
identityѕб)conv2d_transpose_7/BiasAdd/ReadVariableOpб2conv2d_transpose_7/conv2d_transpose/ReadVariableOpN
conv2d_transpose_7/ShapeShapeinputs*
T0*
_output_shapes
:p
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
valueB:░
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
value	B :У
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
valueB:И
"conv2d_transpose_7/strided_slice_1StridedSlice!conv2d_transpose_7/stack:output:01conv2d_transpose_7/strided_slice_1/stack:output:03conv2d_transpose_7/strided_slice_1/stack_1:output:03conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskХ
2conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0Ѓ
#conv2d_transpose_7/conv2d_transposeConv2DBackpropInput!conv2d_transpose_7/stack:output:0:conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ў
)conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0└
conv2d_transpose_7/BiasAddBiasAdd,conv2d_transpose_7/conv2d_transpose:output:01conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ~
conv2d_transpose_7/TanhTanh#conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:         r
IdentityIdentityconv2d_transpose_7/Tanh:y:0^NoOp*
T0*/
_output_shapes
:         Д
NoOpNoOp*^conv2d_transpose_7/BiasAdd/ReadVariableOp3^conv2d_transpose_7/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 2V
)conv2d_transpose_7/BiasAdd/ReadVariableOp)conv2d_transpose_7/BiasAdd/ReadVariableOp2h
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2conv2d_transpose_7/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Є
Я
$__inference_signature_wrapper_388130
input_4"
unknown:	`ђ
	unknown_0:`
	unknown_1:`
	unknown_2:`
	unknown_3:`
	unknown_4:`#
	unknown_5:0`
	unknown_6:0
	unknown_7:0
	unknown_8:0
	unknown_9:0

unknown_10:0$

unknown_11:0

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17:

unknown_18:$

unknown_19:

unknown_20:$

unknown_21:

unknown_22:

unknown_23: $

unknown_24:

unknown_25:
identityѕбStatefulPartitionedCallЌ
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
:         *=
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ **
f%R#
!__inference__wrapped_model_386463w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:         ђ: : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_4
Х
F
*__inference_reshape_1_layer_call_fn_388779

inputs
identity╝
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_387073i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
╠
ю
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_389266

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╩
е
3__inference_conv2d_transpose_7_layer_call_fn_389820

inputs!
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_387045Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
┌ 
ю
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_389607

inputsC
(conv2d_transpose_readvariableop_resource:	`ђ-
biasadd_readvariableop_resource:`
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
valueB:┘
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
valueB:┘
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
value	B :`y
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
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЉ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:	`ђ*
dtype0▄
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           `*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Ў
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           `y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `Ђ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
у
_
C__inference_re_lu_3_layer_call_and_return_conditional_losses_387118

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         `b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         `:W S
/
_output_shapes
:         `
 
_user_specified_nameinputs
Й
D
(__inference_re_lu_4_layer_call_fn_389122

inputs
identity╣
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_re_lu_4_layer_call_and_return_conditional_losses_387163h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         	0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         	0:W S
/
_output_shapes
:         	0
 
_user_specified_nameinputs
╣
╠
:__inference_spectral_normalization_15_layer_call_fn_389488

inputs!
unknown:
	unknown_0:
	unknown_1:
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *^
fYRW
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_387383w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ј
»
:__inference_spectral_normalization_13_layer_call_fn_388969

inputs!
unknown:0`
	unknown_0:0
identityѕбStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *^
fYRW
U__inference_spectral_normalization_13_layer_call_and_return_conditional_losses_387143w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         	0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         `: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         `
 
_user_specified_nameinputs
Є
Ъ
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_386900

inputs)
conv2d_9_386805:
conv2d_9_386807:*
conv2d_10_386821:
conv2d_10_386823:*
conv2d_11_386837:
conv2d_11_386839:%
private__attention_1_386894: 
identity

identity_1ѕб!conv2d_10/StatefulPartitionedCallб!conv2d_11/StatefulPartitionedCallб conv2d_9/StatefulPartitionedCallб,private__attention_1/StatefulPartitionedCallч
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_9_386805conv2d_9_386807*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_386804 
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_386821conv2d_10_386823*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_386820 
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_11_386837conv2d_11_386839*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_386836ф
,private__attention_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*conv2d_10/StatefulPartitionedCall:output:0*conv2d_11/StatefulPartitionedCall:output:0inputsprivate__attention_1_386894*
Tin	
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:         :         DD*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_private__attention_1_layer_call_and_return_conditional_losses_386893ї
IdentityIdentity5private__attention_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         і

Identity_1Identity5private__attention_1/StatefulPartitionedCall:output:1^NoOp*
T0*+
_output_shapes
:         DDЯ
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall-^private__attention_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2\
,private__attention_1/StatefulPartitionedCall,private__attention_1/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Й
D
(__inference_re_lu_3_layer_call_fn_388955

inputs
identity╣
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_387118h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         `:W S
/
_output_shapes
:         `
 
_user_specified_nameinputs
сO
И
C__inference_model_1_layer_call_and_return_conditional_losses_387779

inputs;
 spectral_normalization_12_387700:	`ђ3
 spectral_normalization_12_387702:	ђ.
 spectral_normalization_12_387704:`*
batch_normalization_3_387707:`*
batch_normalization_3_387709:`*
batch_normalization_3_387711:`*
batch_normalization_3_387713:`:
 spectral_normalization_13_387717:0`2
 spectral_normalization_13_387719:`.
 spectral_normalization_13_387721:0*
batch_normalization_4_387724:0*
batch_normalization_4_387726:0*
batch_normalization_4_387728:0*
batch_normalization_4_387730:0:
 spectral_normalization_14_387734:02
 spectral_normalization_14_387736:0.
 spectral_normalization_14_387738:*
batch_normalization_5_387745:*
batch_normalization_5_387747:*
batch_normalization_5_387749:*
batch_normalization_5_387751:2
self_attn_model_1_387755:&
self_attn_model_1_387757:2
self_attn_model_1_387759:&
self_attn_model_1_387761:2
self_attn_model_1_387763:&
self_attn_model_1_387765:"
self_attn_model_1_387767: :
 spectral_normalization_15_387771:2
 spectral_normalization_15_387773:.
 spectral_normalization_15_387775:
identityѕб-batch_normalization_3/StatefulPartitionedCallб-batch_normalization_4/StatefulPartitionedCallб-batch_normalization_5/StatefulPartitionedCallб)self_attn_model_1/StatefulPartitionedCallб1spectral_normalization_12/StatefulPartitionedCallб1spectral_normalization_13/StatefulPartitionedCallб1spectral_normalization_14/StatefulPartitionedCallб1spectral_normalization_15/StatefulPartitionedCallк
reshape_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_387073§
1spectral_normalization_12/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0 spectral_normalization_12_387700 spectral_normalization_12_387702 spectral_normalization_12_387704*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *^
fYRW
U__inference_spectral_normalization_12_layer_call_and_return_conditional_losses_387620А
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_12/StatefulPartitionedCall:output:0batch_normalization_3_387707batch_normalization_3_387709batch_normalization_3_387711batch_normalization_3_387713*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_386560ы
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_387118ч
1spectral_normalization_13/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0 spectral_normalization_13_387717 spectral_normalization_13_387719 spectral_normalization_13_387721*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	0*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *^
fYRW
U__inference_spectral_normalization_13_layer_call_and_return_conditional_losses_387541А
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_13/StatefulPartitionedCall:output:0batch_normalization_4_387724batch_normalization_4_387726batch_normalization_4_387728batch_normalization_4_387730*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_386668ы
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_re_lu_4_layer_call_and_return_conditional_losses_387163ч
1spectral_normalization_14/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0 spectral_normalization_14_387734 spectral_normalization_14_387736 spectral_normalization_14_387738*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *^
fYRW
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_387462Є
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                Ѕ
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               Ѕ
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ї
(tf.__operators__.getitem_1/strided_sliceStridedSlice:spectral_normalization_14/StatefulPartitionedCall:output:07tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_maskў
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall1tf.__operators__.getitem_1/strided_slice:output:0batch_normalization_5_387745batch_normalization_5_387747batch_normalization_5_387749batch_normalization_5_387751*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_386776ы
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_re_lu_5_layer_call_and_return_conditional_losses_387212П
)self_attn_model_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0self_attn_model_1_387755self_attn_model_1_387757self_attn_model_1_387759self_attn_model_1_387761self_attn_model_1_387763self_attn_model_1_387765self_attn_model_1_387767*
Tin

2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:         :         DD*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_386988Ї
1spectral_normalization_15/StatefulPartitionedCallStatefulPartitionedCall2self_attn_model_1/StatefulPartitionedCall:output:0 spectral_normalization_15_387771 spectral_normalization_15_387773 spectral_normalization_15_387775*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *^
fYRW
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_387383Љ
IdentityIdentity:spectral_normalization_15/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         м
NoOpNoOp.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall*^self_attn_model_1/StatefulPartitionedCall2^spectral_normalization_12/StatefulPartitionedCall2^spectral_normalization_13/StatefulPartitionedCall2^spectral_normalization_14/StatefulPartitionedCall2^spectral_normalization_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:         ђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2V
)self_attn_model_1/StatefulPartitionedCall)self_attn_model_1/StatefulPartitionedCall2f
1spectral_normalization_12/StatefulPartitionedCall1spectral_normalization_12/StatefulPartitionedCall2f
1spectral_normalization_13/StatefulPartitionedCall1spectral_normalization_13/StatefulPartitionedCall2f
1spectral_normalization_14/StatefulPartitionedCall1spectral_normalization_14/StatefulPartitionedCall2f
1spectral_normalization_15/StatefulPartitionedCall1spectral_normalization_15/StatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
├
й
5__inference_private__attention_1_layer_call_fn_389703
inputs_0
inputs_1
inputs_2
inputs_3
unknown: 
identity

identity_1ѕбStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3unknown*
Tin	
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:         :         DD*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_private__attention_1_layer_call_and_return_conditional_losses_386893w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         u

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*+
_output_shapes
:         DD`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Ђ
_input_shapesp
n:         :         :         :         : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:         
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/1:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/2:YU
/
_output_shapes
:         
"
_user_specified_name
inputs/3
Н
▄
(__inference_model_1_layer_call_fn_388256

inputs"
unknown:	`ђ
	unknown_0:	ђ
	unknown_1:`
	unknown_2:`
	unknown_3:`
	unknown_4:`
	unknown_5:`#
	unknown_6:0`
	unknown_7:`
	unknown_8:0
	unknown_9:0

unknown_10:0

unknown_11:0

unknown_12:0$

unknown_13:0

unknown_14:0

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:$

unknown_20:

unknown_21:$

unknown_22:

unknown_23:$

unknown_24:

unknown_25:

unknown_26: $

unknown_27:

unknown_28:

unknown_29:
identityѕбStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:         *3
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_387779w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:         ђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
у
_
C__inference_re_lu_5_layer_call_and_return_conditional_losses_389294

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
зя
Х!
!__inference__wrapped_model_386463
input_4x
]model_1_spectral_normalization_12_conv2d_transpose_4_conv2d_transpose_readvariableop_resource:	`ђb
Tmodel_1_spectral_normalization_12_conv2d_transpose_4_biasadd_readvariableop_resource:`C
5model_1_batch_normalization_3_readvariableop_resource:`E
7model_1_batch_normalization_3_readvariableop_1_resource:`T
Fmodel_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:`V
Hmodel_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:`w
]model_1_spectral_normalization_13_conv2d_transpose_5_conv2d_transpose_readvariableop_resource:0`b
Tmodel_1_spectral_normalization_13_conv2d_transpose_5_biasadd_readvariableop_resource:0C
5model_1_batch_normalization_4_readvariableop_resource:0E
7model_1_batch_normalization_4_readvariableop_1_resource:0T
Fmodel_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:0V
Hmodel_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:0w
]model_1_spectral_normalization_14_conv2d_transpose_6_conv2d_transpose_readvariableop_resource:0b
Tmodel_1_spectral_normalization_14_conv2d_transpose_6_biasadd_readvariableop_resource:C
5model_1_batch_normalization_5_readvariableop_resource:E
7model_1_batch_normalization_5_readvariableop_1_resource:T
Fmodel_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:V
Hmodel_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:[
Amodel_1_self_attn_model_1_conv2d_9_conv2d_readvariableop_resource:P
Bmodel_1_self_attn_model_1_conv2d_9_biasadd_readvariableop_resource:\
Bmodel_1_self_attn_model_1_conv2d_10_conv2d_readvariableop_resource:Q
Cmodel_1_self_attn_model_1_conv2d_10_biasadd_readvariableop_resource:\
Bmodel_1_self_attn_model_1_conv2d_11_conv2d_readvariableop_resource:Q
Cmodel_1_self_attn_model_1_conv2d_11_biasadd_readvariableop_resource:V
Lmodel_1_self_attn_model_1_private__attention_1_mul_3_readvariableop_resource: w
]model_1_spectral_normalization_15_conv2d_transpose_7_conv2d_transpose_readvariableop_resource:b
Tmodel_1_spectral_normalization_15_conv2d_transpose_7_biasadd_readvariableop_resource:
identityѕб=model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpб?model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б,model_1/batch_normalization_3/ReadVariableOpб.model_1/batch_normalization_3/ReadVariableOp_1б=model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpб?model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б,model_1/batch_normalization_4/ReadVariableOpб.model_1/batch_normalization_4/ReadVariableOp_1б=model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpб?model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б,model_1/batch_normalization_5/ReadVariableOpб.model_1/batch_normalization_5/ReadVariableOp_1б:model_1/self_attn_model_1/conv2d_10/BiasAdd/ReadVariableOpб9model_1/self_attn_model_1/conv2d_10/Conv2D/ReadVariableOpб:model_1/self_attn_model_1/conv2d_11/BiasAdd/ReadVariableOpб9model_1/self_attn_model_1/conv2d_11/Conv2D/ReadVariableOpб9model_1/self_attn_model_1/conv2d_9/BiasAdd/ReadVariableOpб8model_1/self_attn_model_1/conv2d_9/Conv2D/ReadVariableOpбCmodel_1/self_attn_model_1/private__attention_1/Mul_3/ReadVariableOpбKmodel_1/spectral_normalization_12/conv2d_transpose_4/BiasAdd/ReadVariableOpбTmodel_1/spectral_normalization_12/conv2d_transpose_4/conv2d_transpose/ReadVariableOpбKmodel_1/spectral_normalization_13/conv2d_transpose_5/BiasAdd/ReadVariableOpбTmodel_1/spectral_normalization_13/conv2d_transpose_5/conv2d_transpose/ReadVariableOpбKmodel_1/spectral_normalization_14/conv2d_transpose_6/BiasAdd/ReadVariableOpбTmodel_1/spectral_normalization_14/conv2d_transpose_6/conv2d_transpose/ReadVariableOpбKmodel_1/spectral_normalization_15/conv2d_transpose_7/BiasAdd/ReadVariableOpбTmodel_1/spectral_normalization_15/conv2d_transpose_7/conv2d_transpose/ReadVariableOpN
model_1/reshape_1/ShapeShapeinput_4*
T0*
_output_shapes
:o
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
valueB:Ф
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
value	B :d
!model_1/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :ђЃ
model_1/reshape_1/Reshape/shapePack(model_1/reshape_1/strided_slice:output:0*model_1/reshape_1/Reshape/shape/1:output:0*model_1/reshape_1/Reshape/shape/2:output:0*model_1/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:њ
model_1/reshape_1/ReshapeReshapeinput_4(model_1/reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:         ђї
:model_1/spectral_normalization_12/conv2d_transpose_4/ShapeShape"model_1/reshape_1/Reshape:output:0*
T0*
_output_shapes
:њ
Hmodel_1/spectral_normalization_12/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ћ
Jmodel_1/spectral_normalization_12/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ћ
Jmodel_1/spectral_normalization_12/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┌
Bmodel_1/spectral_normalization_12/conv2d_transpose_4/strided_sliceStridedSliceCmodel_1/spectral_normalization_12/conv2d_transpose_4/Shape:output:0Qmodel_1/spectral_normalization_12/conv2d_transpose_4/strided_slice/stack:output:0Smodel_1/spectral_normalization_12/conv2d_transpose_4/strided_slice/stack_1:output:0Smodel_1/spectral_normalization_12/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
<model_1/spectral_normalization_12/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :~
<model_1/spectral_normalization_12/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :~
<model_1/spectral_normalization_12/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :`њ
:model_1/spectral_normalization_12/conv2d_transpose_4/stackPackKmodel_1/spectral_normalization_12/conv2d_transpose_4/strided_slice:output:0Emodel_1/spectral_normalization_12/conv2d_transpose_4/stack/1:output:0Emodel_1/spectral_normalization_12/conv2d_transpose_4/stack/2:output:0Emodel_1/spectral_normalization_12/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:ћ
Jmodel_1/spectral_normalization_12/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ќ
Lmodel_1/spectral_normalization_12/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ќ
Lmodel_1/spectral_normalization_12/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Р
Dmodel_1/spectral_normalization_12/conv2d_transpose_4/strided_slice_1StridedSliceCmodel_1/spectral_normalization_12/conv2d_transpose_4/stack:output:0Smodel_1/spectral_normalization_12/conv2d_transpose_4/strided_slice_1/stack:output:0Umodel_1/spectral_normalization_12/conv2d_transpose_4/strided_slice_1/stack_1:output:0Umodel_1/spectral_normalization_12/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskч
Tmodel_1/spectral_normalization_12/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp]model_1_spectral_normalization_12_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*'
_output_shapes
:	`ђ*
dtype0Ё
Emodel_1/spectral_normalization_12/conv2d_transpose_4/conv2d_transposeConv2DBackpropInputCmodel_1/spectral_normalization_12/conv2d_transpose_4/stack:output:0\model_1/spectral_normalization_12/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0"model_1/reshape_1/Reshape:output:0*
T0*/
_output_shapes
:         `*
paddingSAME*
strides
▄
Kmodel_1/spectral_normalization_12/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOpTmodel_1_spectral_normalization_12_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0д
<model_1/spectral_normalization_12/conv2d_transpose_4/BiasAddBiasAddNmodel_1/spectral_normalization_12/conv2d_transpose_4/conv2d_transpose:output:0Smodel_1/spectral_normalization_12/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `ъ
,model_1/batch_normalization_3/ReadVariableOpReadVariableOp5model_1_batch_normalization_3_readvariableop_resource*
_output_shapes
:`*
dtype0б
.model_1/batch_normalization_3/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:`*
dtype0└
=model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0─
?model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0І
.model_1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3Emodel_1/spectral_normalization_12/conv2d_transpose_4/BiasAdd:output:04model_1/batch_normalization_3/ReadVariableOp:value:06model_1/batch_normalization_3/ReadVariableOp_1:value:0Emodel_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         `:`:`:`:`:*
epsilon%oЃ:*
is_training( і
model_1/re_lu_3/ReluRelu2model_1/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         `ї
:model_1/spectral_normalization_13/conv2d_transpose_5/ShapeShape"model_1/re_lu_3/Relu:activations:0*
T0*
_output_shapes
:њ
Hmodel_1/spectral_normalization_13/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ћ
Jmodel_1/spectral_normalization_13/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ћ
Jmodel_1/spectral_normalization_13/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┌
Bmodel_1/spectral_normalization_13/conv2d_transpose_5/strided_sliceStridedSliceCmodel_1/spectral_normalization_13/conv2d_transpose_5/Shape:output:0Qmodel_1/spectral_normalization_13/conv2d_transpose_5/strided_slice/stack:output:0Smodel_1/spectral_normalization_13/conv2d_transpose_5/strided_slice/stack_1:output:0Smodel_1/spectral_normalization_13/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
<model_1/spectral_normalization_13/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :~
<model_1/spectral_normalization_13/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :	~
<model_1/spectral_normalization_13/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :0њ
:model_1/spectral_normalization_13/conv2d_transpose_5/stackPackKmodel_1/spectral_normalization_13/conv2d_transpose_5/strided_slice:output:0Emodel_1/spectral_normalization_13/conv2d_transpose_5/stack/1:output:0Emodel_1/spectral_normalization_13/conv2d_transpose_5/stack/2:output:0Emodel_1/spectral_normalization_13/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:ћ
Jmodel_1/spectral_normalization_13/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ќ
Lmodel_1/spectral_normalization_13/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ќ
Lmodel_1/spectral_normalization_13/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Р
Dmodel_1/spectral_normalization_13/conv2d_transpose_5/strided_slice_1StridedSliceCmodel_1/spectral_normalization_13/conv2d_transpose_5/stack:output:0Smodel_1/spectral_normalization_13/conv2d_transpose_5/strided_slice_1/stack:output:0Umodel_1/spectral_normalization_13/conv2d_transpose_5/strided_slice_1/stack_1:output:0Umodel_1/spectral_normalization_13/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЩ
Tmodel_1/spectral_normalization_13/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp]model_1_spectral_normalization_13_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0`*
dtype0Ё
Emodel_1/spectral_normalization_13/conv2d_transpose_5/conv2d_transposeConv2DBackpropInputCmodel_1/spectral_normalization_13/conv2d_transpose_5/stack:output:0\model_1/spectral_normalization_13/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0"model_1/re_lu_3/Relu:activations:0*
T0*/
_output_shapes
:         	0*
paddingSAME*
strides
▄
Kmodel_1/spectral_normalization_13/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOpTmodel_1_spectral_normalization_13_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0д
<model_1/spectral_normalization_13/conv2d_transpose_5/BiasAddBiasAddNmodel_1/spectral_normalization_13/conv2d_transpose_5/conv2d_transpose:output:0Smodel_1/spectral_normalization_13/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         	0ъ
,model_1/batch_normalization_4/ReadVariableOpReadVariableOp5model_1_batch_normalization_4_readvariableop_resource*
_output_shapes
:0*
dtype0б
.model_1/batch_normalization_4/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:0*
dtype0└
=model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0─
?model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0І
.model_1/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3Emodel_1/spectral_normalization_13/conv2d_transpose_5/BiasAdd:output:04model_1/batch_normalization_4/ReadVariableOp:value:06model_1/batch_normalization_4/ReadVariableOp_1:value:0Emodel_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         	0:0:0:0:0:*
epsilon%oЃ:*
is_training( і
model_1/re_lu_4/ReluRelu2model_1/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         	0ї
:model_1/spectral_normalization_14/conv2d_transpose_6/ShapeShape"model_1/re_lu_4/Relu:activations:0*
T0*
_output_shapes
:њ
Hmodel_1/spectral_normalization_14/conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ћ
Jmodel_1/spectral_normalization_14/conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ћ
Jmodel_1/spectral_normalization_14/conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┌
Bmodel_1/spectral_normalization_14/conv2d_transpose_6/strided_sliceStridedSliceCmodel_1/spectral_normalization_14/conv2d_transpose_6/Shape:output:0Qmodel_1/spectral_normalization_14/conv2d_transpose_6/strided_slice/stack:output:0Smodel_1/spectral_normalization_14/conv2d_transpose_6/strided_slice/stack_1:output:0Smodel_1/spectral_normalization_14/conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
<model_1/spectral_normalization_14/conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :~
<model_1/spectral_normalization_14/conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :~
<model_1/spectral_normalization_14/conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :њ
:model_1/spectral_normalization_14/conv2d_transpose_6/stackPackKmodel_1/spectral_normalization_14/conv2d_transpose_6/strided_slice:output:0Emodel_1/spectral_normalization_14/conv2d_transpose_6/stack/1:output:0Emodel_1/spectral_normalization_14/conv2d_transpose_6/stack/2:output:0Emodel_1/spectral_normalization_14/conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:ћ
Jmodel_1/spectral_normalization_14/conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ќ
Lmodel_1/spectral_normalization_14/conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ќ
Lmodel_1/spectral_normalization_14/conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Р
Dmodel_1/spectral_normalization_14/conv2d_transpose_6/strided_slice_1StridedSliceCmodel_1/spectral_normalization_14/conv2d_transpose_6/stack:output:0Smodel_1/spectral_normalization_14/conv2d_transpose_6/strided_slice_1/stack:output:0Umodel_1/spectral_normalization_14/conv2d_transpose_6/strided_slice_1/stack_1:output:0Umodel_1/spectral_normalization_14/conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЩ
Tmodel_1/spectral_normalization_14/conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOp]model_1_spectral_normalization_14_conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0*
dtype0Ё
Emodel_1/spectral_normalization_14/conv2d_transpose_6/conv2d_transposeConv2DBackpropInputCmodel_1/spectral_normalization_14/conv2d_transpose_6/stack:output:0\model_1/spectral_normalization_14/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0"model_1/re_lu_4/Relu:activations:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
▄
Kmodel_1/spectral_normalization_14/conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOpTmodel_1_spectral_normalization_14_conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0д
<model_1/spectral_normalization_14/conv2d_transpose_6/BiasAddBiasAddNmodel_1/spectral_normalization_14/conv2d_transpose_6/conv2d_transpose:output:0Smodel_1/spectral_normalization_14/conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         Ј
6model_1/tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                Љ
8model_1/tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               Љ
8model_1/tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            и
0model_1/tf.__operators__.getitem_1/strided_sliceStridedSliceEmodel_1/spectral_normalization_14/conv2d_transpose_6/BiasAdd:output:0?model_1/tf.__operators__.getitem_1/strided_slice/stack:output:0Amodel_1/tf.__operators__.getitem_1/strided_slice/stack_1:output:0Amodel_1/tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_maskъ
,model_1/batch_normalization_5/ReadVariableOpReadVariableOp5model_1_batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype0б
.model_1/batch_normalization_5/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype0└
=model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0─
?model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0 
.model_1/batch_normalization_5/FusedBatchNormV3FusedBatchNormV39model_1/tf.__operators__.getitem_1/strided_slice:output:04model_1/batch_normalization_5/ReadVariableOp:value:06model_1/batch_normalization_5/ReadVariableOp_1:value:0Emodel_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oЃ:*
is_training( і
model_1/re_lu_5/ReluRelu2model_1/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         ┬
8model_1/self_attn_model_1/conv2d_9/Conv2D/ReadVariableOpReadVariableOpAmodel_1_self_attn_model_1_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ч
)model_1/self_attn_model_1/conv2d_9/Conv2DConv2D"model_1/re_lu_5/Relu:activations:0@model_1/self_attn_model_1/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
И
9model_1/self_attn_model_1/conv2d_9/BiasAdd/ReadVariableOpReadVariableOpBmodel_1_self_attn_model_1_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Т
*model_1/self_attn_model_1/conv2d_9/BiasAddBiasAdd2model_1/self_attn_model_1/conv2d_9/Conv2D:output:0Amodel_1/self_attn_model_1/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ─
9model_1/self_attn_model_1/conv2d_10/Conv2D/ReadVariableOpReadVariableOpBmodel_1_self_attn_model_1_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0■
*model_1/self_attn_model_1/conv2d_10/Conv2DConv2D"model_1/re_lu_5/Relu:activations:0Amodel_1/self_attn_model_1/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
║
:model_1/self_attn_model_1/conv2d_10/BiasAdd/ReadVariableOpReadVariableOpCmodel_1_self_attn_model_1_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ж
+model_1/self_attn_model_1/conv2d_10/BiasAddBiasAdd3model_1/self_attn_model_1/conv2d_10/Conv2D:output:0Bmodel_1/self_attn_model_1/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ─
9model_1/self_attn_model_1/conv2d_11/Conv2D/ReadVariableOpReadVariableOpBmodel_1_self_attn_model_1_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0■
*model_1/self_attn_model_1/conv2d_11/Conv2DConv2D"model_1/re_lu_5/Relu:activations:0Amodel_1/self_attn_model_1/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
║
:model_1/self_attn_model_1/conv2d_11/BiasAdd/ReadVariableOpReadVariableOpCmodel_1_self_attn_model_1_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ж
+model_1/self_attn_model_1/conv2d_11/BiasAddBiasAdd3model_1/self_attn_model_1/conv2d_11/Conv2D:output:0Bmodel_1/self_attn_model_1/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         Ќ
4model_1/self_attn_model_1/private__attention_1/ShapeShape3model_1/self_attn_model_1/conv2d_9/BiasAdd:output:0*
T0*
_output_shapes
:ї
Bmodel_1/self_attn_model_1/private__attention_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ј
Dmodel_1/self_attn_model_1/private__attention_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ј
Dmodel_1/self_attn_model_1/private__attention_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╝
<model_1/self_attn_model_1/private__attention_1/strided_sliceStridedSlice=model_1/self_attn_model_1/private__attention_1/Shape:output:0Kmodel_1/self_attn_model_1/private__attention_1/strided_slice/stack:output:0Mmodel_1/self_attn_model_1/private__attention_1/strided_slice/stack_1:output:0Mmodel_1/self_attn_model_1/private__attention_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskј
Dmodel_1/self_attn_model_1/private__attention_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:љ
Fmodel_1/self_attn_model_1/private__attention_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:љ
Fmodel_1/self_attn_model_1/private__attention_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:─
>model_1/self_attn_model_1/private__attention_1/strided_slice_1StridedSlice=model_1/self_attn_model_1/private__attention_1/Shape:output:0Mmodel_1/self_attn_model_1/private__attention_1/strided_slice_1/stack:output:0Omodel_1/self_attn_model_1/private__attention_1/strided_slice_1/stack_1:output:0Omodel_1/self_attn_model_1/private__attention_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskј
Dmodel_1/self_attn_model_1/private__attention_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:љ
Fmodel_1/self_attn_model_1/private__attention_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:љ
Fmodel_1/self_attn_model_1/private__attention_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:─
>model_1/self_attn_model_1/private__attention_1/strided_slice_2StridedSlice=model_1/self_attn_model_1/private__attention_1/Shape:output:0Mmodel_1/self_attn_model_1/private__attention_1/strided_slice_2/stack:output:0Omodel_1/self_attn_model_1/private__attention_1/strided_slice_2/stack_1:output:0Omodel_1/self_attn_model_1/private__attention_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskВ
2model_1/self_attn_model_1/private__attention_1/mulMulGmodel_1/self_attn_model_1/private__attention_1/strided_slice_1:output:0Gmodel_1/self_attn_model_1/private__attention_1/strided_slice_2:output:0*
T0*
_output_shapes
: Ѕ
>model_1/self_attn_model_1/private__attention_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
         ║
<model_1/self_attn_model_1/private__attention_1/Reshape/shapePackEmodel_1/self_attn_model_1/private__attention_1/strided_slice:output:06model_1/self_attn_model_1/private__attention_1/mul:z:0Gmodel_1/self_attn_model_1/private__attention_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:Ч
6model_1/self_attn_model_1/private__attention_1/ReshapeReshape3model_1/self_attn_model_1/conv2d_9/BiasAdd:output:0Emodel_1/self_attn_model_1/private__attention_1/Reshape/shape:output:0*
T0*4
_output_shapes"
 :         D         Ь
4model_1/self_attn_model_1/private__attention_1/mul_1MulGmodel_1/self_attn_model_1/private__attention_1/strided_slice_1:output:0Gmodel_1/self_attn_model_1/private__attention_1/strided_slice_2:output:0*
T0*
_output_shapes
: І
@model_1/self_attn_model_1/private__attention_1/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
         └
>model_1/self_attn_model_1/private__attention_1/Reshape_1/shapePackEmodel_1/self_attn_model_1/private__attention_1/strided_slice:output:08model_1/self_attn_model_1/private__attention_1/mul_1:z:0Imodel_1/self_attn_model_1/private__attention_1/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:Ђ
8model_1/self_attn_model_1/private__attention_1/Reshape_1Reshape4model_1/self_attn_model_1/conv2d_10/BiasAdd:output:0Gmodel_1/self_attn_model_1/private__attention_1/Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :         D         њ
=model_1/self_attn_model_1/private__attention_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ј
8model_1/self_attn_model_1/private__attention_1/transpose	TransposeAmodel_1/self_attn_model_1/private__attention_1/Reshape_1:output:0Fmodel_1/self_attn_model_1/private__attention_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :                  DЬ
4model_1/self_attn_model_1/private__attention_1/mul_2MulGmodel_1/self_attn_model_1/private__attention_1/strided_slice_1:output:0Gmodel_1/self_attn_model_1/private__attention_1/strided_slice_2:output:0*
T0*
_output_shapes
: І
@model_1/self_attn_model_1/private__attention_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
         └
>model_1/self_attn_model_1/private__attention_1/Reshape_2/shapePackEmodel_1/self_attn_model_1/private__attention_1/strided_slice:output:08model_1/self_attn_model_1/private__attention_1/mul_2:z:0Imodel_1/self_attn_model_1/private__attention_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:Ђ
8model_1/self_attn_model_1/private__attention_1/Reshape_2Reshape4model_1/self_attn_model_1/conv2d_11/BiasAdd:output:0Gmodel_1/self_attn_model_1/private__attention_1/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :         D         ћ
?model_1/self_attn_model_1/private__attention_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Њ
:model_1/self_attn_model_1/private__attention_1/transpose_1	TransposeAmodel_1/self_attn_model_1/private__attention_1/Reshape_2:output:0Hmodel_1/self_attn_model_1/private__attention_1/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  Dч
5model_1/self_attn_model_1/private__attention_1/MatMulBatchMatMulV2?model_1/self_attn_model_1/private__attention_1/Reshape:output:0<model_1/self_attn_model_1/private__attention_1/transpose:y:0*
T0*+
_output_shapes
:         DDи
6model_1/self_attn_model_1/private__attention_1/SoftmaxSoftmax>model_1/self_attn_model_1/private__attention_1/MatMul:output:0*
T0*+
_output_shapes
:         DDћ
?model_1/self_attn_model_1/private__attention_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ѕ
:model_1/self_attn_model_1/private__attention_1/transpose_2	Transpose@model_1/self_attn_model_1/private__attention_1/Softmax:softmax:0Hmodel_1/self_attn_model_1/private__attention_1/transpose_2/perm:output:0*
T0*+
_output_shapes
:         DDЄ
7model_1/self_attn_model_1/private__attention_1/MatMul_1BatchMatMulV2>model_1/self_attn_model_1/private__attention_1/transpose_1:y:0>model_1/self_attn_model_1/private__attention_1/transpose_2:y:0*
T0*4
_output_shapes"
 :                  Dћ
?model_1/self_attn_model_1/private__attention_1/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          њ
:model_1/self_attn_model_1/private__attention_1/transpose_3	Transpose@model_1/self_attn_model_1/private__attention_1/MatMul_1:output:0Hmodel_1/self_attn_model_1/private__attention_1/transpose_3/perm:output:0*
T0*4
_output_shapes"
 :         D         І
@model_1/self_attn_model_1/private__attention_1/Reshape_3/shape/3Const*
_output_shapes
: *
dtype0*
valueB :
         ў
>model_1/self_attn_model_1/private__attention_1/Reshape_3/shapePackEmodel_1/self_attn_model_1/private__attention_1/strided_slice:output:0Gmodel_1/self_attn_model_1/private__attention_1/strided_slice_1:output:0Gmodel_1/self_attn_model_1/private__attention_1/strided_slice_2:output:0Imodel_1/self_attn_model_1/private__attention_1/Reshape_3/shape/3:output:0*
N*
T0*
_output_shapes
:Ј
8model_1/self_attn_model_1/private__attention_1/Reshape_3Reshape>model_1/self_attn_model_1/private__attention_1/transpose_3:y:0Gmodel_1/self_attn_model_1/private__attention_1/Reshape_3/shape:output:0*
T0*8
_output_shapes&
$:"                  ╚
Cmodel_1/self_attn_model_1/private__attention_1/Mul_3/ReadVariableOpReadVariableOpLmodel_1_self_attn_model_1_private__attention_1_mul_3_readvariableop_resource*
_output_shapes
: *
dtype0ј
4model_1/self_attn_model_1/private__attention_1/Mul_3MulAmodel_1/self_attn_model_1/private__attention_1/Reshape_3:output:0Kmodel_1/self_attn_model_1/private__attention_1/Mul_3/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  М
2model_1/self_attn_model_1/private__attention_1/AddAddV28model_1/self_attn_model_1/private__attention_1/Mul_3:z:0"model_1/re_lu_5/Relu:activations:0*
T0*/
_output_shapes
:         а
:model_1/spectral_normalization_15/conv2d_transpose_7/ShapeShape6model_1/self_attn_model_1/private__attention_1/Add:z:0*
T0*
_output_shapes
:њ
Hmodel_1/spectral_normalization_15/conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ћ
Jmodel_1/spectral_normalization_15/conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ћ
Jmodel_1/spectral_normalization_15/conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┌
Bmodel_1/spectral_normalization_15/conv2d_transpose_7/strided_sliceStridedSliceCmodel_1/spectral_normalization_15/conv2d_transpose_7/Shape:output:0Qmodel_1/spectral_normalization_15/conv2d_transpose_7/strided_slice/stack:output:0Smodel_1/spectral_normalization_15/conv2d_transpose_7/strided_slice/stack_1:output:0Smodel_1/spectral_normalization_15/conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
<model_1/spectral_normalization_15/conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :~
<model_1/spectral_normalization_15/conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :~
<model_1/spectral_normalization_15/conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :њ
:model_1/spectral_normalization_15/conv2d_transpose_7/stackPackKmodel_1/spectral_normalization_15/conv2d_transpose_7/strided_slice:output:0Emodel_1/spectral_normalization_15/conv2d_transpose_7/stack/1:output:0Emodel_1/spectral_normalization_15/conv2d_transpose_7/stack/2:output:0Emodel_1/spectral_normalization_15/conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:ћ
Jmodel_1/spectral_normalization_15/conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ќ
Lmodel_1/spectral_normalization_15/conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ќ
Lmodel_1/spectral_normalization_15/conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Р
Dmodel_1/spectral_normalization_15/conv2d_transpose_7/strided_slice_1StridedSliceCmodel_1/spectral_normalization_15/conv2d_transpose_7/stack:output:0Smodel_1/spectral_normalization_15/conv2d_transpose_7/strided_slice_1/stack:output:0Umodel_1/spectral_normalization_15/conv2d_transpose_7/strided_slice_1/stack_1:output:0Umodel_1/spectral_normalization_15/conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЩ
Tmodel_1/spectral_normalization_15/conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOp]model_1_spectral_normalization_15_conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0Ў
Emodel_1/spectral_normalization_15/conv2d_transpose_7/conv2d_transposeConv2DBackpropInputCmodel_1/spectral_normalization_15/conv2d_transpose_7/stack:output:0\model_1/spectral_normalization_15/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:06model_1/self_attn_model_1/private__attention_1/Add:z:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
▄
Kmodel_1/spectral_normalization_15/conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOpTmodel_1_spectral_normalization_15_conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0д
<model_1/spectral_normalization_15/conv2d_transpose_7/BiasAddBiasAddNmodel_1/spectral_normalization_15/conv2d_transpose_7/conv2d_transpose:output:0Smodel_1/spectral_normalization_15/conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ┬
9model_1/spectral_normalization_15/conv2d_transpose_7/TanhTanhEmodel_1/spectral_normalization_15/conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:         ћ
IdentityIdentity=model_1/spectral_normalization_15/conv2d_transpose_7/Tanh:y:0^NoOp*
T0*/
_output_shapes
:         »
NoOpNoOp>^model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp@^model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1-^model_1/batch_normalization_3/ReadVariableOp/^model_1/batch_normalization_3/ReadVariableOp_1>^model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp@^model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1-^model_1/batch_normalization_4/ReadVariableOp/^model_1/batch_normalization_4/ReadVariableOp_1>^model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp@^model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1-^model_1/batch_normalization_5/ReadVariableOp/^model_1/batch_normalization_5/ReadVariableOp_1;^model_1/self_attn_model_1/conv2d_10/BiasAdd/ReadVariableOp:^model_1/self_attn_model_1/conv2d_10/Conv2D/ReadVariableOp;^model_1/self_attn_model_1/conv2d_11/BiasAdd/ReadVariableOp:^model_1/self_attn_model_1/conv2d_11/Conv2D/ReadVariableOp:^model_1/self_attn_model_1/conv2d_9/BiasAdd/ReadVariableOp9^model_1/self_attn_model_1/conv2d_9/Conv2D/ReadVariableOpD^model_1/self_attn_model_1/private__attention_1/Mul_3/ReadVariableOpL^model_1/spectral_normalization_12/conv2d_transpose_4/BiasAdd/ReadVariableOpU^model_1/spectral_normalization_12/conv2d_transpose_4/conv2d_transpose/ReadVariableOpL^model_1/spectral_normalization_13/conv2d_transpose_5/BiasAdd/ReadVariableOpU^model_1/spectral_normalization_13/conv2d_transpose_5/conv2d_transpose/ReadVariableOpL^model_1/spectral_normalization_14/conv2d_transpose_6/BiasAdd/ReadVariableOpU^model_1/spectral_normalization_14/conv2d_transpose_6/conv2d_transpose/ReadVariableOpL^model_1/spectral_normalization_15/conv2d_transpose_7/BiasAdd/ReadVariableOpU^model_1/spectral_normalization_15/conv2d_transpose_7/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:         ђ: : : : : : : : : : : : : : : : : : : : : : : : : : : 2~
=model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp=model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2ѓ
?model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?model_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12\
,model_1/batch_normalization_3/ReadVariableOp,model_1/batch_normalization_3/ReadVariableOp2`
.model_1/batch_normalization_3/ReadVariableOp_1.model_1/batch_normalization_3/ReadVariableOp_12~
=model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp=model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2ѓ
?model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?model_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12\
,model_1/batch_normalization_4/ReadVariableOp,model_1/batch_normalization_4/ReadVariableOp2`
.model_1/batch_normalization_4/ReadVariableOp_1.model_1/batch_normalization_4/ReadVariableOp_12~
=model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp=model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2ѓ
?model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?model_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12\
,model_1/batch_normalization_5/ReadVariableOp,model_1/batch_normalization_5/ReadVariableOp2`
.model_1/batch_normalization_5/ReadVariableOp_1.model_1/batch_normalization_5/ReadVariableOp_12x
:model_1/self_attn_model_1/conv2d_10/BiasAdd/ReadVariableOp:model_1/self_attn_model_1/conv2d_10/BiasAdd/ReadVariableOp2v
9model_1/self_attn_model_1/conv2d_10/Conv2D/ReadVariableOp9model_1/self_attn_model_1/conv2d_10/Conv2D/ReadVariableOp2x
:model_1/self_attn_model_1/conv2d_11/BiasAdd/ReadVariableOp:model_1/self_attn_model_1/conv2d_11/BiasAdd/ReadVariableOp2v
9model_1/self_attn_model_1/conv2d_11/Conv2D/ReadVariableOp9model_1/self_attn_model_1/conv2d_11/Conv2D/ReadVariableOp2v
9model_1/self_attn_model_1/conv2d_9/BiasAdd/ReadVariableOp9model_1/self_attn_model_1/conv2d_9/BiasAdd/ReadVariableOp2t
8model_1/self_attn_model_1/conv2d_9/Conv2D/ReadVariableOp8model_1/self_attn_model_1/conv2d_9/Conv2D/ReadVariableOp2і
Cmodel_1/self_attn_model_1/private__attention_1/Mul_3/ReadVariableOpCmodel_1/self_attn_model_1/private__attention_1/Mul_3/ReadVariableOp2џ
Kmodel_1/spectral_normalization_12/conv2d_transpose_4/BiasAdd/ReadVariableOpKmodel_1/spectral_normalization_12/conv2d_transpose_4/BiasAdd/ReadVariableOp2г
Tmodel_1/spectral_normalization_12/conv2d_transpose_4/conv2d_transpose/ReadVariableOpTmodel_1/spectral_normalization_12/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2џ
Kmodel_1/spectral_normalization_13/conv2d_transpose_5/BiasAdd/ReadVariableOpKmodel_1/spectral_normalization_13/conv2d_transpose_5/BiasAdd/ReadVariableOp2г
Tmodel_1/spectral_normalization_13/conv2d_transpose_5/conv2d_transpose/ReadVariableOpTmodel_1/spectral_normalization_13/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2џ
Kmodel_1/spectral_normalization_14/conv2d_transpose_6/BiasAdd/ReadVariableOpKmodel_1/spectral_normalization_14/conv2d_transpose_6/BiasAdd/ReadVariableOp2г
Tmodel_1/spectral_normalization_14/conv2d_transpose_6/conv2d_transpose/ReadVariableOpTmodel_1/spectral_normalization_14/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2џ
Kmodel_1/spectral_normalization_15/conv2d_transpose_7/BiasAdd/ReadVariableOpKmodel_1/spectral_normalization_15/conv2d_transpose_7/BiasAdd/ReadVariableOp2г
Tmodel_1/spectral_normalization_15/conv2d_transpose_7/conv2d_transpose/ReadVariableOpTmodel_1/spectral_normalization_15/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_4
■ђ
 
"__inference__traced_restore_390073
file_prefixL
1assignvariableop_spectral_normalization_12_kernel:	`ђD
1assignvariableop_1_spectral_normalization_12_sn_u:	ђ<
.assignvariableop_2_batch_normalization_3_gamma:`;
-assignvariableop_3_batch_normalization_3_beta:`B
4assignvariableop_4_batch_normalization_3_moving_mean:`F
8assignvariableop_5_batch_normalization_3_moving_variance:`M
3assignvariableop_6_spectral_normalization_13_kernel:0`C
1assignvariableop_7_spectral_normalization_13_sn_u:`<
.assignvariableop_8_batch_normalization_4_gamma:0;
-assignvariableop_9_batch_normalization_4_beta:0C
5assignvariableop_10_batch_normalization_4_moving_mean:0G
9assignvariableop_11_batch_normalization_4_moving_variance:0N
4assignvariableop_12_spectral_normalization_14_kernel:0D
2assignvariableop_13_spectral_normalization_14_sn_u:0=
/assignvariableop_14_batch_normalization_5_gamma:<
.assignvariableop_15_batch_normalization_5_beta:C
5assignvariableop_16_batch_normalization_5_moving_mean:G
9assignvariableop_17_batch_normalization_5_moving_variance:N
4assignvariableop_18_spectral_normalization_15_kernel:D
2assignvariableop_19_spectral_normalization_15_sn_u:@
2assignvariableop_20_spectral_normalization_12_bias:`@
2assignvariableop_21_spectral_normalization_13_bias:0@
2assignvariableop_22_spectral_normalization_14_bias:_
Uassignvariableop_23_self_attn_model_1_private__attention_1_private__attention_1_gamma: O
5assignvariableop_24_self_attn_model_1_conv2d_9_kernel:A
3assignvariableop_25_self_attn_model_1_conv2d_9_bias:P
6assignvariableop_26_self_attn_model_1_conv2d_10_kernel:B
4assignvariableop_27_self_attn_model_1_conv2d_10_bias:P
6assignvariableop_28_self_attn_model_1_conv2d_11_kernel:B
4assignvariableop_29_self_attn_model_1_conv2d_11_bias:@
2assignvariableop_30_spectral_normalization_15_bias:
identity_32ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9│
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*┘
value¤B╠ B1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/sn_u/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-2/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/sn_u/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-4/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/sn_u/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-7/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/sn_u/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH░
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ┴
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ќ
_output_shapesЃ
ђ::::::::::::::::::::::::::::::::*.
dtypes$
"2 [
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOpAssignVariableOp1assignvariableop_spectral_normalization_12_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_1AssignVariableOp1assignvariableop_1_spectral_normalization_12_sn_uIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_3_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_3_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_3_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_3_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_6AssignVariableOp3assignvariableop_6_spectral_normalization_13_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_7AssignVariableOp1assignvariableop_7_spectral_normalization_13_sn_uIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_4_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_4_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_4_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_4_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_12AssignVariableOp4assignvariableop_12_spectral_normalization_14_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_13AssignVariableOp2assignvariableop_13_spectral_normalization_14_sn_uIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_5_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_5_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_5_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_5_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_18AssignVariableOp4assignvariableop_18_spectral_normalization_15_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_19AssignVariableOp2assignvariableop_19_spectral_normalization_15_sn_uIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_20AssignVariableOp2assignvariableop_20_spectral_normalization_12_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_21AssignVariableOp2assignvariableop_21_spectral_normalization_13_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_22AssignVariableOp2assignvariableop_22_spectral_normalization_14_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_23AssignVariableOpUassignvariableop_23_self_attn_model_1_private__attention_1_private__attention_1_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_24AssignVariableOp5assignvariableop_24_self_attn_model_1_conv2d_9_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_25AssignVariableOp3assignvariableop_25_self_attn_model_1_conv2d_9_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_26AssignVariableOp6assignvariableop_26_self_attn_model_1_conv2d_10_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_27AssignVariableOp4assignvariableop_27_self_attn_model_1_conv2d_10_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_28AssignVariableOp6assignvariableop_28_self_attn_model_1_conv2d_11_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_29AssignVariableOp4assignvariableop_29_self_attn_model_1_conv2d_11_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_30AssignVariableOp2assignvariableop_30_spectral_normalization_15_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 щ
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: Т
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_32Identity_32:output:0*S
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
_user_specified_namefile_prefix
┌ 
ю
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_386500

inputsC
(conv2d_transpose_readvariableop_resource:	`ђ-
biasadd_readvariableop_resource:`
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
valueB:┘
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
valueB:┘
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
value	B :`y
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
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЉ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:	`ђ*
dtype0▄
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           `*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Ў
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           `y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `Ђ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Є
Ъ
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_386988

inputs)
conv2d_9_386967:
conv2d_9_386969:*
conv2d_10_386972:
conv2d_10_386974:*
conv2d_11_386977:
conv2d_11_386979:%
private__attention_1_386982: 
identity

identity_1ѕб!conv2d_10/StatefulPartitionedCallб!conv2d_11/StatefulPartitionedCallб conv2d_9/StatefulPartitionedCallб,private__attention_1/StatefulPartitionedCallч
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_9_386967conv2d_9_386969*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_386804 
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_386972conv2d_10_386974*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_386820 
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_11_386977conv2d_11_386979*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_386836ф
,private__attention_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*conv2d_10/StatefulPartitionedCall:output:0*conv2d_11/StatefulPartitionedCall:output:0inputsprivate__attention_1_386982*
Tin	
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:         :         DD*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_private__attention_1_layer_call_and_return_conditional_losses_386893ї
IdentityIdentity5private__attention_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         і

Identity_1Identity5private__attention_1/StatefulPartitionedCall:output:1^NoOp*
T0*+
_output_shapes
:         DDЯ
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall-^private__attention_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2\
,private__attention_1/StatefulPartitionedCall,private__attention_1/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
нG
═
U__inference_spectral_normalization_13_layer_call_and_return_conditional_losses_389055

inputs9
reshape_readvariableop_resource:0`C
1spectral_normalize_matmul_readvariableop_resource:`@
2conv2d_transpose_5_biasadd_readvariableop_resource:0
identityѕбReshape/ReadVariableOpб)conv2d_transpose_5/BiasAdd/ReadVariableOpб2conv2d_transpose_5/conv2d_transpose/ReadVariableOpб#spectral_normalize/AssignVariableOpб%spectral_normalize/AssignVariableOp_1б(spectral_normalize/MatMul/ReadVariableOpб!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:0`*
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   t
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	Я`џ
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:`*
dtype0ц
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes
:	Я*
transpose_b(
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	Яv
%spectral_normalize/l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       └
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
 *╠╝ї+й
'spectral_normalize/l2_normalize/MaximumMaximum,spectral_normalize/l2_normalize/Sum:output:02spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:ё
%spectral_normalize/l2_normalize/RsqrtRsqrt+spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:а
spectral_normalize/l2_normalizeMul#spectral_normalize/MatMul:product:0)spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes
:	ЯЁ
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes

:`ѓ
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:`x
'spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       к
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
 *╠╝ї+├
)spectral_normalize/l2_normalize_1/MaximumMaximum.spectral_normalize/l2_normalize_1/Sum:output:04spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:ѕ
'spectral_normalize/l2_normalize_1/RsqrtRsqrt-spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:Ц
!spectral_normalize/l2_normalize_1Mul%spectral_normalize/MatMul_1:product:0+spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

:`
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:`ђ
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	Яї
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes

:`▓
spectral_normalize/MatMul_3MatMul%spectral_normalize/MatMul_2:product:0(spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(Э
#spectral_normalize/AssignVariableOpAssignVariableOp1spectral_normalize_matmul_readvariableop_resource(spectral_normalize/StopGradient:output:0)^spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ѕ
!spectral_normalize/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:0`*
dtype0е
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:0`y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      0   `   А
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:0`ш
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(N
conv2d_transpose_5/ShapeShapeinputs*
T0*
_output_shapes
:p
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
valueB:░
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
value	B :0У
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
valueB:И
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask┬
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
:0`*
dtype0Ѓ
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:         	0*
paddingSAME*
strides
ў
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0└
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         	0z
IdentityIdentity#conv2d_transpose_5/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         	0П
NoOpNoOp^Reshape/ReadVariableOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         `: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:         `
 
_user_specified_nameinputs
╣
╠
:__inference_spectral_normalization_13_layer_call_fn_388980

inputs!
unknown:0`
	unknown_0:`
	unknown_1:0
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	0*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *^
fYRW
U__inference_spectral_normalization_13_layer_call_and_return_conditional_losses_387541w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         	0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         `: : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         `
 
_user_specified_nameinputs
ЃH
Л
__inference__traced_save_389970
file_prefix?
;savev2_spectral_normalization_12_kernel_read_readvariableop=
9savev2_spectral_normalization_12_sn_u_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop?
;savev2_spectral_normalization_13_kernel_read_readvariableop=
9savev2_spectral_normalization_13_sn_u_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop?
;savev2_spectral_normalization_14_kernel_read_readvariableop=
9savev2_spectral_normalization_14_sn_u_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop?
;savev2_spectral_normalization_15_kernel_read_readvariableop=
9savev2_spectral_normalization_15_sn_u_read_readvariableop=
9savev2_spectral_normalization_12_bias_read_readvariableop=
9savev2_spectral_normalization_13_bias_read_readvariableop=
9savev2_spectral_normalization_14_bias_read_readvariableop`
\savev2_self_attn_model_1_private__attention_1_private__attention_1_gamma_read_readvariableop@
<savev2_self_attn_model_1_conv2d_9_kernel_read_readvariableop>
:savev2_self_attn_model_1_conv2d_9_bias_read_readvariableopA
=savev2_self_attn_model_1_conv2d_10_kernel_read_readvariableop?
;savev2_self_attn_model_1_conv2d_10_bias_read_readvariableopA
=savev2_self_attn_model_1_conv2d_11_kernel_read_readvariableop?
;savev2_self_attn_model_1_conv2d_11_bias_read_readvariableop=
9savev2_spectral_normalization_15_bias_read_readvariableop
savev2_const

identity_1ѕбMergeV2Checkpointsw
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
_temp/partЂ
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
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ░
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*┘
value¤B╠ B1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/sn_u/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-2/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/sn_u/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-4/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/sn_u/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-7/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/sn_u/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHГ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B И
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0;savev2_spectral_normalization_12_kernel_read_readvariableop9savev2_spectral_normalization_12_sn_u_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop;savev2_spectral_normalization_13_kernel_read_readvariableop9savev2_spectral_normalization_13_sn_u_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop;savev2_spectral_normalization_14_kernel_read_readvariableop9savev2_spectral_normalization_14_sn_u_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop;savev2_spectral_normalization_15_kernel_read_readvariableop9savev2_spectral_normalization_15_sn_u_read_readvariableop9savev2_spectral_normalization_12_bias_read_readvariableop9savev2_spectral_normalization_13_bias_read_readvariableop9savev2_spectral_normalization_14_bias_read_readvariableop\savev2_self_attn_model_1_private__attention_1_private__attention_1_gamma_read_readvariableop<savev2_self_attn_model_1_conv2d_9_kernel_read_readvariableop:savev2_self_attn_model_1_conv2d_9_bias_read_readvariableop=savev2_self_attn_model_1_conv2d_10_kernel_read_readvariableop;savev2_self_attn_model_1_conv2d_10_bias_read_readvariableop=savev2_self_attn_model_1_conv2d_11_kernel_read_readvariableop;savev2_self_attn_model_1_conv2d_11_bias_read_readvariableop9savev2_spectral_normalization_15_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:І
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

identity_1Identity_1:output:0*х
_input_shapesБ
а: :	`ђ:	ђ:`:`:`:`:0`:`:0:0:0:0:0:0:::::::`:0:: :::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:	`ђ:%!

_output_shapes
:	ђ: 

_output_shapes
:`: 

_output_shapes
:`: 

_output_shapes
:`: 

_output_shapes
:`:,(
&
_output_shapes
:0`:$ 

_output_shapes

:`: 	

_output_shapes
:0: 


_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0:,(
&
_output_shapes
:0:$ 

_output_shapes

:0: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:`: 

_output_shapes
:0: 

_output_shapes
::

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
: 
Е

■
E__inference_conv2d_10_layer_call_and_return_conditional_losses_389792

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╠
ю
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_388932

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           `:`:`:`:`:*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           `░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           `: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           `
 
_user_specified_nameinputs
═
№
U__inference_spectral_normalization_12_layer_call_and_return_conditional_losses_387098

inputsV
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource:	`ђ@
2conv2d_transpose_4_biasadd_readvariableop_resource:`
identityѕб)conv2d_transpose_4/BiasAdd/ReadVariableOpб2conv2d_transpose_4/conv2d_transpose/ReadVariableOpN
conv2d_transpose_4/ShapeShapeinputs*
T0*
_output_shapes
:p
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
valueB:░
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
value	B :\
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :`У
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
valueB:И
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskи
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*'
_output_shapes
:	`ђ*
dtype0Ѓ
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:         `*
paddingSAME*
strides
ў
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0└
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `z
IdentityIdentity#conv2d_transpose_4/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         `Д
NoOpNoOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
╠
ю
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_386637

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           0:0:0:0:0:*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           0░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           0
 
_user_specified_nameinputs
Е

■
E__inference_conv2d_11_layer_call_and_return_conditional_losses_389811

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ІM
У
C__inference_model_1_layer_call_and_return_conditional_losses_387986
input_4;
 spectral_normalization_12_387915:	`ђ.
 spectral_normalization_12_387917:`*
batch_normalization_3_387920:`*
batch_normalization_3_387922:`*
batch_normalization_3_387924:`*
batch_normalization_3_387926:`:
 spectral_normalization_13_387930:0`.
 spectral_normalization_13_387932:0*
batch_normalization_4_387935:0*
batch_normalization_4_387937:0*
batch_normalization_4_387939:0*
batch_normalization_4_387941:0:
 spectral_normalization_14_387945:0.
 spectral_normalization_14_387947:*
batch_normalization_5_387954:*
batch_normalization_5_387956:*
batch_normalization_5_387958:*
batch_normalization_5_387960:2
self_attn_model_1_387964:&
self_attn_model_1_387966:2
self_attn_model_1_387968:&
self_attn_model_1_387970:2
self_attn_model_1_387972:&
self_attn_model_1_387974:"
self_attn_model_1_387976: :
 spectral_normalization_15_387980:.
 spectral_normalization_15_387982:
identityѕб-batch_normalization_3/StatefulPartitionedCallб-batch_normalization_4/StatefulPartitionedCallб-batch_normalization_5/StatefulPartitionedCallб)self_attn_model_1/StatefulPartitionedCallб1spectral_normalization_12/StatefulPartitionedCallб1spectral_normalization_13/StatefulPartitionedCallб1spectral_normalization_14/StatefulPartitionedCallб1spectral_normalization_15/StatefulPartitionedCallК
reshape_1/PartitionedCallPartitionedCallinput_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_387073█
1spectral_normalization_12/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0 spectral_normalization_12_387915 spectral_normalization_12_387917*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *^
fYRW
U__inference_spectral_normalization_12_layer_call_and_return_conditional_losses_387098Б
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_12/StatefulPartitionedCall:output:0batch_normalization_3_387920batch_normalization_3_387922batch_normalization_3_387924batch_normalization_3_387926*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_386529ы
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_387118┘
1spectral_normalization_13/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0 spectral_normalization_13_387930 spectral_normalization_13_387932*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *^
fYRW
U__inference_spectral_normalization_13_layer_call_and_return_conditional_losses_387143Б
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_13/StatefulPartitionedCall:output:0batch_normalization_4_387935batch_normalization_4_387937batch_normalization_4_387939batch_normalization_4_387941*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_386637ы
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         	0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_re_lu_4_layer_call_and_return_conditional_losses_387163┘
1spectral_normalization_14/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0 spectral_normalization_14_387945 spectral_normalization_14_387947*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *^
fYRW
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_387188Є
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                Ѕ
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               Ѕ
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ї
(tf.__operators__.getitem_1/strided_sliceStridedSlice:spectral_normalization_14/StatefulPartitionedCall:output:07tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_maskџ
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall1tf.__operators__.getitem_1/strided_slice:output:0batch_normalization_5_387954batch_normalization_5_387956batch_normalization_5_387958batch_normalization_5_387960*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_386745ы
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_re_lu_5_layer_call_and_return_conditional_losses_387212П
)self_attn_model_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0self_attn_model_1_387964self_attn_model_1_387966self_attn_model_1_387968self_attn_model_1_387970self_attn_model_1_387972self_attn_model_1_387974self_attn_model_1_387976*
Tin

2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:         :         DD*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_386900в
1spectral_normalization_15/StatefulPartitionedCallStatefulPartitionedCall2self_attn_model_1/StatefulPartitionedCall:output:0 spectral_normalization_15_387980 spectral_normalization_15_387982*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *^
fYRW
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_387254Љ
IdentityIdentity:spectral_normalization_15/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         м
NoOpNoOp.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall*^self_attn_model_1/StatefulPartitionedCall2^spectral_normalization_12/StatefulPartitionedCall2^spectral_normalization_13/StatefulPartitionedCall2^spectral_normalization_14/StatefulPartitionedCall2^spectral_normalization_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:         ђ: : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2V
)self_attn_model_1/StatefulPartitionedCall)self_attn_model_1/StatefulPartitionedCall2f
1spectral_normalization_12/StatefulPartitionedCall1spectral_normalization_12/StatefulPartitionedCall2f
1spectral_normalization_13/StatefulPartitionedCall1spectral_normalization_13/StatefulPartitionedCall2f
1spectral_normalization_14/StatefulPartitionedCall1spectral_normalization_14/StatefulPartitionedCall2f
1spectral_normalization_15/StatefulPartitionedCall1spectral_normalization_15/StatefulPartitionedCall:Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_4
н/
ж
P__inference_private__attention_1_layer_call_and_return_conditional_losses_386893

inputs
inputs_1
inputs_2
inputs_3'
mul_3_readvariableop_resource: 
identity

identity_1ѕбMul_3/ReadVariableOp;
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
valueB:Л
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
valueB:┘
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
valueB:┘
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
         ~
Reshape/shapePackstrided_slice:output:0mul:z:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:q
ReshapeReshapeinputsReshape/shape:output:0*
T0*4
_output_shapes"
 :         D         a
mul_1Mulstrided_slice_1:output:0strided_slice_2:output:0*
T0*
_output_shapes
: \
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
         ё
Reshape_1/shapePackstrided_slice:output:0	mul_1:z:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:w
	Reshape_1Reshapeinputs_1Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :         D         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ѓ
	transpose	TransposeReshape_1:output:0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  Da
mul_2Mulstrided_slice_1:output:0strided_slice_2:output:0*
T0*
_output_shapes
: \
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
         ё
Reshape_2/shapePackstrided_slice:output:0	mul_2:z:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:w
	Reshape_2Reshapeinputs_2Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :         D         e
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          є
transpose_1	TransposeReshape_2:output:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  Dn
MatMulBatchMatMulV2Reshape:output:0transpose:y:0*
T0*+
_output_shapes
:         DDY
SoftmaxSoftmaxMatMul:output:0*
T0*+
_output_shapes
:         DDe
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_2	TransposeSoftmax:softmax:0transpose_2/perm:output:0*
T0*+
_output_shapes
:         DDz
MatMul_1BatchMatMulV2transpose_1:y:0transpose_2:y:0*
T0*4
_output_shapes"
 :                  De
transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ё
transpose_3	TransposeMatMul_1:output:0transpose_3/perm:output:0*
T0*4
_output_shapes"
 :         D         \
Reshape_3/shape/3Const*
_output_shapes
: *
dtype0*
valueB :
         Г
Reshape_3/shapePackstrided_slice:output:0strided_slice_1:output:0strided_slice_2:output:0Reshape_3/shape/3:output:0*
N*
T0*
_output_shapes
:ѓ
	Reshape_3Reshapetranspose_3:y:0Reshape_3/shape:output:0*
T0*8
_output_shapes&
$:"                  j
Mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes
: *
dtype0Ђ
Mul_3MulReshape_3:output:0Mul_3/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  [
AddAddV2	Mul_3:z:0inputs_3*
T0*/
_output_shapes
:         ^
IdentityIdentityAdd:z:0^NoOp*
T0*/
_output_shapes
:         f

Identity_1IdentitySoftmax:softmax:0^NoOp*
T0*+
_output_shapes
:         DD]
NoOpNoOp^Mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Ђ
_input_shapesp
n:         :         :         :         : 2,
Mul_3/ReadVariableOpMul_3/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs:WS
/
_output_shapes
:         
 
_user_specified_nameinputs
╩
е
3__inference_conv2d_transpose_6_layer_call_fn_389658

inputs!
unknown:0
	unknown_0:
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_386716Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           0: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           0
 
_user_specified_nameinputs
е

§
D__inference_conv2d_9_layer_call_and_return_conditional_losses_389773

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
є
└
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_386668

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           0:0:0:0:0:*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           0н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           0
 
_user_specified_nameinputs
┴
Ь
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_389512

inputsU
;conv2d_transpose_7_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_7_biasadd_readvariableop_resource:
identityѕб)conv2d_transpose_7/BiasAdd/ReadVariableOpб2conv2d_transpose_7/conv2d_transpose/ReadVariableOpN
conv2d_transpose_7/ShapeShapeinputs*
T0*
_output_shapes
:p
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
valueB:░
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
value	B :У
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
valueB:И
"conv2d_transpose_7/strided_slice_1StridedSlice!conv2d_transpose_7/stack:output:01conv2d_transpose_7/strided_slice_1/stack:output:03conv2d_transpose_7/strided_slice_1/stack_1:output:03conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskХ
2conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0Ѓ
#conv2d_transpose_7/conv2d_transposeConv2DBackpropInput!conv2d_transpose_7/stack:output:0:conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ў
)conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0└
conv2d_transpose_7/BiasAddBiasAdd,conv2d_transpose_7/conv2d_transpose:output:01conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ~
conv2d_transpose_7/TanhTanh#conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:         r
IdentityIdentityconv2d_transpose_7/Tanh:y:0^NoOp*
T0*/
_output_shapes
:         Д
NoOpNoOp*^conv2d_transpose_7/BiasAdd/ReadVariableOp3^conv2d_transpose_7/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 2V
)conv2d_transpose_7/BiasAdd/ReadVariableOp)conv2d_transpose_7/BiasAdd/ReadVariableOp2h
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2conv2d_transpose_7/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
»└
Д
C__inference_model_1_layer_call_and_return_conditional_losses_388457

inputsp
Uspectral_normalization_12_conv2d_transpose_4_conv2d_transpose_readvariableop_resource:	`ђZ
Lspectral_normalization_12_conv2d_transpose_4_biasadd_readvariableop_resource:`;
-batch_normalization_3_readvariableop_resource:`=
/batch_normalization_3_readvariableop_1_resource:`L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:`N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:`o
Uspectral_normalization_13_conv2d_transpose_5_conv2d_transpose_readvariableop_resource:0`Z
Lspectral_normalization_13_conv2d_transpose_5_biasadd_readvariableop_resource:0;
-batch_normalization_4_readvariableop_resource:0=
/batch_normalization_4_readvariableop_1_resource:0L
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:0N
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:0o
Uspectral_normalization_14_conv2d_transpose_6_conv2d_transpose_readvariableop_resource:0Z
Lspectral_normalization_14_conv2d_transpose_6_biasadd_readvariableop_resource:;
-batch_normalization_5_readvariableop_resource:=
/batch_normalization_5_readvariableop_1_resource:L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:S
9self_attn_model_1_conv2d_9_conv2d_readvariableop_resource:H
:self_attn_model_1_conv2d_9_biasadd_readvariableop_resource:T
:self_attn_model_1_conv2d_10_conv2d_readvariableop_resource:I
;self_attn_model_1_conv2d_10_biasadd_readvariableop_resource:T
:self_attn_model_1_conv2d_11_conv2d_readvariableop_resource:I
;self_attn_model_1_conv2d_11_biasadd_readvariableop_resource:N
Dself_attn_model_1_private__attention_1_mul_3_readvariableop_resource: o
Uspectral_normalization_15_conv2d_transpose_7_conv2d_transpose_readvariableop_resource:Z
Lspectral_normalization_15_conv2d_transpose_7_biasadd_readvariableop_resource:
identityѕб5batch_normalization_3/FusedBatchNormV3/ReadVariableOpб7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_3/ReadVariableOpб&batch_normalization_3/ReadVariableOp_1б5batch_normalization_4/FusedBatchNormV3/ReadVariableOpб7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_4/ReadVariableOpб&batch_normalization_4/ReadVariableOp_1б5batch_normalization_5/FusedBatchNormV3/ReadVariableOpб7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_5/ReadVariableOpб&batch_normalization_5/ReadVariableOp_1б2self_attn_model_1/conv2d_10/BiasAdd/ReadVariableOpб1self_attn_model_1/conv2d_10/Conv2D/ReadVariableOpб2self_attn_model_1/conv2d_11/BiasAdd/ReadVariableOpб1self_attn_model_1/conv2d_11/Conv2D/ReadVariableOpб1self_attn_model_1/conv2d_9/BiasAdd/ReadVariableOpб0self_attn_model_1/conv2d_9/Conv2D/ReadVariableOpб;self_attn_model_1/private__attention_1/Mul_3/ReadVariableOpбCspectral_normalization_12/conv2d_transpose_4/BiasAdd/ReadVariableOpбLspectral_normalization_12/conv2d_transpose_4/conv2d_transpose/ReadVariableOpбCspectral_normalization_13/conv2d_transpose_5/BiasAdd/ReadVariableOpбLspectral_normalization_13/conv2d_transpose_5/conv2d_transpose/ReadVariableOpбCspectral_normalization_14/conv2d_transpose_6/BiasAdd/ReadVariableOpбLspectral_normalization_14/conv2d_transpose_6/conv2d_transpose/ReadVariableOpбCspectral_normalization_15/conv2d_transpose_7/BiasAdd/ReadVariableOpбLspectral_normalization_15/conv2d_transpose_7/conv2d_transpose/ReadVariableOpE
reshape_1/ShapeShapeinputs*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :\
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :ђ█
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ђ
reshape_1/ReshapeReshapeinputs reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:         ђ|
2spectral_normalization_12/conv2d_transpose_4/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
:і
@spectral_normalization_12/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ї
Bspectral_normalization_12/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ї
Bspectral_normalization_12/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▓
:spectral_normalization_12/conv2d_transpose_4/strided_sliceStridedSlice;spectral_normalization_12/conv2d_transpose_4/Shape:output:0Ispectral_normalization_12/conv2d_transpose_4/strided_slice/stack:output:0Kspectral_normalization_12/conv2d_transpose_4/strided_slice/stack_1:output:0Kspectral_normalization_12/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4spectral_normalization_12/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :v
4spectral_normalization_12/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :v
4spectral_normalization_12/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :`Ж
2spectral_normalization_12/conv2d_transpose_4/stackPackCspectral_normalization_12/conv2d_transpose_4/strided_slice:output:0=spectral_normalization_12/conv2d_transpose_4/stack/1:output:0=spectral_normalization_12/conv2d_transpose_4/stack/2:output:0=spectral_normalization_12/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:ї
Bspectral_normalization_12/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ј
Dspectral_normalization_12/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ј
Dspectral_normalization_12/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
<spectral_normalization_12/conv2d_transpose_4/strided_slice_1StridedSlice;spectral_normalization_12/conv2d_transpose_4/stack:output:0Kspectral_normalization_12/conv2d_transpose_4/strided_slice_1/stack:output:0Mspectral_normalization_12/conv2d_transpose_4/strided_slice_1/stack_1:output:0Mspectral_normalization_12/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskв
Lspectral_normalization_12/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpUspectral_normalization_12_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*'
_output_shapes
:	`ђ*
dtype0т
=spectral_normalization_12/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput;spectral_normalization_12/conv2d_transpose_4/stack:output:0Tspectral_normalization_12/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0reshape_1/Reshape:output:0*
T0*/
_output_shapes
:         `*
paddingSAME*
strides
╠
Cspectral_normalization_12/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOpLspectral_normalization_12_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0ј
4spectral_normalization_12/conv2d_transpose_4/BiasAddBiasAddFspectral_normalization_12/conv2d_transpose_4/conv2d_transpose:output:0Kspectral_normalization_12/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         `ј
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:`*
dtype0њ
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:`*
dtype0░
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0┤
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0█
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3=spectral_normalization_12/conv2d_transpose_4/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         `:`:`:`:`:*
epsilon%oЃ:*
is_training( z
re_lu_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         `|
2spectral_normalization_13/conv2d_transpose_5/ShapeShapere_lu_3/Relu:activations:0*
T0*
_output_shapes
:і
@spectral_normalization_13/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ї
Bspectral_normalization_13/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ї
Bspectral_normalization_13/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▓
:spectral_normalization_13/conv2d_transpose_5/strided_sliceStridedSlice;spectral_normalization_13/conv2d_transpose_5/Shape:output:0Ispectral_normalization_13/conv2d_transpose_5/strided_slice/stack:output:0Kspectral_normalization_13/conv2d_transpose_5/strided_slice/stack_1:output:0Kspectral_normalization_13/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4spectral_normalization_13/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :v
4spectral_normalization_13/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :	v
4spectral_normalization_13/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :0Ж
2spectral_normalization_13/conv2d_transpose_5/stackPackCspectral_normalization_13/conv2d_transpose_5/strided_slice:output:0=spectral_normalization_13/conv2d_transpose_5/stack/1:output:0=spectral_normalization_13/conv2d_transpose_5/stack/2:output:0=spectral_normalization_13/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:ї
Bspectral_normalization_13/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ј
Dspectral_normalization_13/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ј
Dspectral_normalization_13/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
<spectral_normalization_13/conv2d_transpose_5/strided_slice_1StridedSlice;spectral_normalization_13/conv2d_transpose_5/stack:output:0Kspectral_normalization_13/conv2d_transpose_5/strided_slice_1/stack:output:0Mspectral_normalization_13/conv2d_transpose_5/strided_slice_1/stack_1:output:0Mspectral_normalization_13/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЖ
Lspectral_normalization_13/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpUspectral_normalization_13_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0`*
dtype0т
=spectral_normalization_13/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput;spectral_normalization_13/conv2d_transpose_5/stack:output:0Tspectral_normalization_13/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0re_lu_3/Relu:activations:0*
T0*/
_output_shapes
:         	0*
paddingSAME*
strides
╠
Cspectral_normalization_13/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOpLspectral_normalization_13_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0ј
4spectral_normalization_13/conv2d_transpose_5/BiasAddBiasAddFspectral_normalization_13/conv2d_transpose_5/conv2d_transpose:output:0Kspectral_normalization_13/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         	0ј
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:0*
dtype0њ
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:0*
dtype0░
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0┤
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0█
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3=spectral_normalization_13/conv2d_transpose_5/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         	0:0:0:0:0:*
epsilon%oЃ:*
is_training( z
re_lu_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         	0|
2spectral_normalization_14/conv2d_transpose_6/ShapeShapere_lu_4/Relu:activations:0*
T0*
_output_shapes
:і
@spectral_normalization_14/conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ї
Bspectral_normalization_14/conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ї
Bspectral_normalization_14/conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▓
:spectral_normalization_14/conv2d_transpose_6/strided_sliceStridedSlice;spectral_normalization_14/conv2d_transpose_6/Shape:output:0Ispectral_normalization_14/conv2d_transpose_6/strided_slice/stack:output:0Kspectral_normalization_14/conv2d_transpose_6/strided_slice/stack_1:output:0Kspectral_normalization_14/conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4spectral_normalization_14/conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :v
4spectral_normalization_14/conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :v
4spectral_normalization_14/conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :Ж
2spectral_normalization_14/conv2d_transpose_6/stackPackCspectral_normalization_14/conv2d_transpose_6/strided_slice:output:0=spectral_normalization_14/conv2d_transpose_6/stack/1:output:0=spectral_normalization_14/conv2d_transpose_6/stack/2:output:0=spectral_normalization_14/conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:ї
Bspectral_normalization_14/conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ј
Dspectral_normalization_14/conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ј
Dspectral_normalization_14/conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
<spectral_normalization_14/conv2d_transpose_6/strided_slice_1StridedSlice;spectral_normalization_14/conv2d_transpose_6/stack:output:0Kspectral_normalization_14/conv2d_transpose_6/strided_slice_1/stack:output:0Mspectral_normalization_14/conv2d_transpose_6/strided_slice_1/stack_1:output:0Mspectral_normalization_14/conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЖ
Lspectral_normalization_14/conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOpUspectral_normalization_14_conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0*
dtype0т
=spectral_normalization_14/conv2d_transpose_6/conv2d_transposeConv2DBackpropInput;spectral_normalization_14/conv2d_transpose_6/stack:output:0Tspectral_normalization_14/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0re_lu_4/Relu:activations:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
╠
Cspectral_normalization_14/conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOpLspectral_normalization_14_conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ј
4spectral_normalization_14/conv2d_transpose_6/BiasAddBiasAddFspectral_normalization_14/conv2d_transpose_6/conv2d_transpose:output:0Kspectral_normalization_14/conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         Є
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                Ѕ
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               Ѕ
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ј
(tf.__operators__.getitem_1/strided_sliceStridedSlice=spectral_normalization_14/conv2d_transpose_6/BiasAdd:output:07tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_maskј
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype0њ
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype0░
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0┤
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0¤
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV31tf.__operators__.getitem_1/strided_slice:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oЃ:*
is_training( z
re_lu_5/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         ▓
0self_attn_model_1/conv2d_9/Conv2D/ReadVariableOpReadVariableOp9self_attn_model_1_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0С
!self_attn_model_1/conv2d_9/Conv2DConv2Dre_lu_5/Relu:activations:08self_attn_model_1/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
е
1self_attn_model_1/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp:self_attn_model_1_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╬
"self_attn_model_1/conv2d_9/BiasAddBiasAdd*self_attn_model_1/conv2d_9/Conv2D:output:09self_attn_model_1/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ┤
1self_attn_model_1/conv2d_10/Conv2D/ReadVariableOpReadVariableOp:self_attn_model_1_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Т
"self_attn_model_1/conv2d_10/Conv2DConv2Dre_lu_5/Relu:activations:09self_attn_model_1/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
ф
2self_attn_model_1/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp;self_attn_model_1_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
#self_attn_model_1/conv2d_10/BiasAddBiasAdd+self_attn_model_1/conv2d_10/Conv2D:output:0:self_attn_model_1/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ┤
1self_attn_model_1/conv2d_11/Conv2D/ReadVariableOpReadVariableOp:self_attn_model_1_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Т
"self_attn_model_1/conv2d_11/Conv2DConv2Dre_lu_5/Relu:activations:09self_attn_model_1/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
ф
2self_attn_model_1/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp;self_attn_model_1_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
#self_attn_model_1/conv2d_11/BiasAddBiasAdd+self_attn_model_1/conv2d_11/Conv2D:output:0:self_attn_model_1/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         Є
,self_attn_model_1/private__attention_1/ShapeShape+self_attn_model_1/conv2d_9/BiasAdd:output:0*
T0*
_output_shapes
:ё
:self_attn_model_1/private__attention_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: є
<self_attn_model_1/private__attention_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:є
<self_attn_model_1/private__attention_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ћ
4self_attn_model_1/private__attention_1/strided_sliceStridedSlice5self_attn_model_1/private__attention_1/Shape:output:0Cself_attn_model_1/private__attention_1/strided_slice/stack:output:0Eself_attn_model_1/private__attention_1/strided_slice/stack_1:output:0Eself_attn_model_1/private__attention_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskє
<self_attn_model_1/private__attention_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:ѕ
>self_attn_model_1/private__attention_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ѕ
>self_attn_model_1/private__attention_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
6self_attn_model_1/private__attention_1/strided_slice_1StridedSlice5self_attn_model_1/private__attention_1/Shape:output:0Eself_attn_model_1/private__attention_1/strided_slice_1/stack:output:0Gself_attn_model_1/private__attention_1/strided_slice_1/stack_1:output:0Gself_attn_model_1/private__attention_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskє
<self_attn_model_1/private__attention_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:ѕ
>self_attn_model_1/private__attention_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ѕ
>self_attn_model_1/private__attention_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
6self_attn_model_1/private__attention_1/strided_slice_2StridedSlice5self_attn_model_1/private__attention_1/Shape:output:0Eself_attn_model_1/private__attention_1/strided_slice_2/stack:output:0Gself_attn_model_1/private__attention_1/strided_slice_2/stack_1:output:0Gself_attn_model_1/private__attention_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskн
*self_attn_model_1/private__attention_1/mulMul?self_attn_model_1/private__attention_1/strided_slice_1:output:0?self_attn_model_1/private__attention_1/strided_slice_2:output:0*
T0*
_output_shapes
: Ђ
6self_attn_model_1/private__attention_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
         џ
4self_attn_model_1/private__attention_1/Reshape/shapePack=self_attn_model_1/private__attention_1/strided_slice:output:0.self_attn_model_1/private__attention_1/mul:z:0?self_attn_model_1/private__attention_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:С
.self_attn_model_1/private__attention_1/ReshapeReshape+self_attn_model_1/conv2d_9/BiasAdd:output:0=self_attn_model_1/private__attention_1/Reshape/shape:output:0*
T0*4
_output_shapes"
 :         D         о
,self_attn_model_1/private__attention_1/mul_1Mul?self_attn_model_1/private__attention_1/strided_slice_1:output:0?self_attn_model_1/private__attention_1/strided_slice_2:output:0*
T0*
_output_shapes
: Ѓ
8self_attn_model_1/private__attention_1/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
         а
6self_attn_model_1/private__attention_1/Reshape_1/shapePack=self_attn_model_1/private__attention_1/strided_slice:output:00self_attn_model_1/private__attention_1/mul_1:z:0Aself_attn_model_1/private__attention_1/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:ж
0self_attn_model_1/private__attention_1/Reshape_1Reshape,self_attn_model_1/conv2d_10/BiasAdd:output:0?self_attn_model_1/private__attention_1/Reshape_1/shape:output:0*
T0*4
_output_shapes"
 :         D         і
5self_attn_model_1/private__attention_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          э
0self_attn_model_1/private__attention_1/transpose	Transpose9self_attn_model_1/private__attention_1/Reshape_1:output:0>self_attn_model_1/private__attention_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :                  Dо
,self_attn_model_1/private__attention_1/mul_2Mul?self_attn_model_1/private__attention_1/strided_slice_1:output:0?self_attn_model_1/private__attention_1/strided_slice_2:output:0*
T0*
_output_shapes
: Ѓ
8self_attn_model_1/private__attention_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
         а
6self_attn_model_1/private__attention_1/Reshape_2/shapePack=self_attn_model_1/private__attention_1/strided_slice:output:00self_attn_model_1/private__attention_1/mul_2:z:0Aself_attn_model_1/private__attention_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:ж
0self_attn_model_1/private__attention_1/Reshape_2Reshape,self_attn_model_1/conv2d_11/BiasAdd:output:0?self_attn_model_1/private__attention_1/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :         D         ї
7self_attn_model_1/private__attention_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ч
2self_attn_model_1/private__attention_1/transpose_1	Transpose9self_attn_model_1/private__attention_1/Reshape_2:output:0@self_attn_model_1/private__attention_1/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  Dс
-self_attn_model_1/private__attention_1/MatMulBatchMatMulV27self_attn_model_1/private__attention_1/Reshape:output:04self_attn_model_1/private__attention_1/transpose:y:0*
T0*+
_output_shapes
:         DDД
.self_attn_model_1/private__attention_1/SoftmaxSoftmax6self_attn_model_1/private__attention_1/MatMul:output:0*
T0*+
_output_shapes
:         DDї
7self_attn_model_1/private__attention_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ы
2self_attn_model_1/private__attention_1/transpose_2	Transpose8self_attn_model_1/private__attention_1/Softmax:softmax:0@self_attn_model_1/private__attention_1/transpose_2/perm:output:0*
T0*+
_output_shapes
:         DD№
/self_attn_model_1/private__attention_1/MatMul_1BatchMatMulV26self_attn_model_1/private__attention_1/transpose_1:y:06self_attn_model_1/private__attention_1/transpose_2:y:0*
T0*4
_output_shapes"
 :                  Dї
7self_attn_model_1/private__attention_1/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          Щ
2self_attn_model_1/private__attention_1/transpose_3	Transpose8self_attn_model_1/private__attention_1/MatMul_1:output:0@self_attn_model_1/private__attention_1/transpose_3/perm:output:0*
T0*4
_output_shapes"
 :         D         Ѓ
8self_attn_model_1/private__attention_1/Reshape_3/shape/3Const*
_output_shapes
: *
dtype0*
valueB :
         ­
6self_attn_model_1/private__attention_1/Reshape_3/shapePack=self_attn_model_1/private__attention_1/strided_slice:output:0?self_attn_model_1/private__attention_1/strided_slice_1:output:0?self_attn_model_1/private__attention_1/strided_slice_2:output:0Aself_attn_model_1/private__attention_1/Reshape_3/shape/3:output:0*
N*
T0*
_output_shapes
:э
0self_attn_model_1/private__attention_1/Reshape_3Reshape6self_attn_model_1/private__attention_1/transpose_3:y:0?self_attn_model_1/private__attention_1/Reshape_3/shape:output:0*
T0*8
_output_shapes&
$:"                  И
;self_attn_model_1/private__attention_1/Mul_3/ReadVariableOpReadVariableOpDself_attn_model_1_private__attention_1_mul_3_readvariableop_resource*
_output_shapes
: *
dtype0Ш
,self_attn_model_1/private__attention_1/Mul_3Mul9self_attn_model_1/private__attention_1/Reshape_3:output:0Cself_attn_model_1/private__attention_1/Mul_3/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  ╗
*self_attn_model_1/private__attention_1/AddAddV20self_attn_model_1/private__attention_1/Mul_3:z:0re_lu_5/Relu:activations:0*
T0*/
_output_shapes
:         љ
2spectral_normalization_15/conv2d_transpose_7/ShapeShape.self_attn_model_1/private__attention_1/Add:z:0*
T0*
_output_shapes
:і
@spectral_normalization_15/conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ї
Bspectral_normalization_15/conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ї
Bspectral_normalization_15/conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▓
:spectral_normalization_15/conv2d_transpose_7/strided_sliceStridedSlice;spectral_normalization_15/conv2d_transpose_7/Shape:output:0Ispectral_normalization_15/conv2d_transpose_7/strided_slice/stack:output:0Kspectral_normalization_15/conv2d_transpose_7/strided_slice/stack_1:output:0Kspectral_normalization_15/conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4spectral_normalization_15/conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :v
4spectral_normalization_15/conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :v
4spectral_normalization_15/conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :Ж
2spectral_normalization_15/conv2d_transpose_7/stackPackCspectral_normalization_15/conv2d_transpose_7/strided_slice:output:0=spectral_normalization_15/conv2d_transpose_7/stack/1:output:0=spectral_normalization_15/conv2d_transpose_7/stack/2:output:0=spectral_normalization_15/conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:ї
Bspectral_normalization_15/conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ј
Dspectral_normalization_15/conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ј
Dspectral_normalization_15/conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
<spectral_normalization_15/conv2d_transpose_7/strided_slice_1StridedSlice;spectral_normalization_15/conv2d_transpose_7/stack:output:0Kspectral_normalization_15/conv2d_transpose_7/strided_slice_1/stack:output:0Mspectral_normalization_15/conv2d_transpose_7/strided_slice_1/stack_1:output:0Mspectral_normalization_15/conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЖ
Lspectral_normalization_15/conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOpUspectral_normalization_15_conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0щ
=spectral_normalization_15/conv2d_transpose_7/conv2d_transposeConv2DBackpropInput;spectral_normalization_15/conv2d_transpose_7/stack:output:0Tspectral_normalization_15/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0.self_attn_model_1/private__attention_1/Add:z:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
╠
Cspectral_normalization_15/conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOpLspectral_normalization_15_conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ј
4spectral_normalization_15/conv2d_transpose_7/BiasAddBiasAddFspectral_normalization_15/conv2d_transpose_7/conv2d_transpose:output:0Kspectral_normalization_15/conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ▓
1spectral_normalization_15/conv2d_transpose_7/TanhTanh=spectral_normalization_15/conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:         ї
IdentityIdentity5spectral_normalization_15/conv2d_transpose_7/Tanh:y:0^NoOp*
T0*/
_output_shapes
:         О
NoOpNoOp6^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_13^self_attn_model_1/conv2d_10/BiasAdd/ReadVariableOp2^self_attn_model_1/conv2d_10/Conv2D/ReadVariableOp3^self_attn_model_1/conv2d_11/BiasAdd/ReadVariableOp2^self_attn_model_1/conv2d_11/Conv2D/ReadVariableOp2^self_attn_model_1/conv2d_9/BiasAdd/ReadVariableOp1^self_attn_model_1/conv2d_9/Conv2D/ReadVariableOp<^self_attn_model_1/private__attention_1/Mul_3/ReadVariableOpD^spectral_normalization_12/conv2d_transpose_4/BiasAdd/ReadVariableOpM^spectral_normalization_12/conv2d_transpose_4/conv2d_transpose/ReadVariableOpD^spectral_normalization_13/conv2d_transpose_5/BiasAdd/ReadVariableOpM^spectral_normalization_13/conv2d_transpose_5/conv2d_transpose/ReadVariableOpD^spectral_normalization_14/conv2d_transpose_6/BiasAdd/ReadVariableOpM^spectral_normalization_14/conv2d_transpose_6/conv2d_transpose/ReadVariableOpD^spectral_normalization_15/conv2d_transpose_7/BiasAdd/ReadVariableOpM^spectral_normalization_15/conv2d_transpose_7/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:         ђ: : : : : : : : : : : : : : : : : : : : : : : : : : : 2n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12h
2self_attn_model_1/conv2d_10/BiasAdd/ReadVariableOp2self_attn_model_1/conv2d_10/BiasAdd/ReadVariableOp2f
1self_attn_model_1/conv2d_10/Conv2D/ReadVariableOp1self_attn_model_1/conv2d_10/Conv2D/ReadVariableOp2h
2self_attn_model_1/conv2d_11/BiasAdd/ReadVariableOp2self_attn_model_1/conv2d_11/BiasAdd/ReadVariableOp2f
1self_attn_model_1/conv2d_11/Conv2D/ReadVariableOp1self_attn_model_1/conv2d_11/Conv2D/ReadVariableOp2f
1self_attn_model_1/conv2d_9/BiasAdd/ReadVariableOp1self_attn_model_1/conv2d_9/BiasAdd/ReadVariableOp2d
0self_attn_model_1/conv2d_9/Conv2D/ReadVariableOp0self_attn_model_1/conv2d_9/Conv2D/ReadVariableOp2z
;self_attn_model_1/private__attention_1/Mul_3/ReadVariableOp;self_attn_model_1/private__attention_1/Mul_3/ReadVariableOp2і
Cspectral_normalization_12/conv2d_transpose_4/BiasAdd/ReadVariableOpCspectral_normalization_12/conv2d_transpose_4/BiasAdd/ReadVariableOp2ю
Lspectral_normalization_12/conv2d_transpose_4/conv2d_transpose/ReadVariableOpLspectral_normalization_12/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2і
Cspectral_normalization_13/conv2d_transpose_5/BiasAdd/ReadVariableOpCspectral_normalization_13/conv2d_transpose_5/BiasAdd/ReadVariableOp2ю
Lspectral_normalization_13/conv2d_transpose_5/conv2d_transpose/ReadVariableOpLspectral_normalization_13/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2і
Cspectral_normalization_14/conv2d_transpose_6/BiasAdd/ReadVariableOpCspectral_normalization_14/conv2d_transpose_6/BiasAdd/ReadVariableOp2ю
Lspectral_normalization_14/conv2d_transpose_6/conv2d_transpose/ReadVariableOpLspectral_normalization_14/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2і
Cspectral_normalization_15/conv2d_transpose_7/BiasAdd/ReadVariableOpCspectral_normalization_15/conv2d_transpose_7/BiasAdd/ReadVariableOp2ю
Lspectral_normalization_15/conv2d_transpose_7/conv2d_transpose/ReadVariableOpLspectral_normalization_15/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
╬G
═
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_387462

inputs9
reshape_readvariableop_resource:0C
1spectral_normalize_matmul_readvariableop_resource:0@
2conv2d_transpose_6_biasadd_readvariableop_resource:
identityѕбReshape/ReadVariableOpб)conv2d_transpose_6/BiasAdd/ReadVariableOpб2conv2d_transpose_6/conv2d_transpose/ReadVariableOpб#spectral_normalize/AssignVariableOpб%spectral_normalize/AssignVariableOp_1б(spectral_normalize/MatMul/ReadVariableOpб!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:0*
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    0   s
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes

:H0џ
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:0*
dtype0Б
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
valueB"       └
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
 *╠╝ї+й
'spectral_normalize/l2_normalize/MaximumMaximum,spectral_normalize/l2_normalize/Sum:output:02spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:ё
%spectral_normalize/l2_normalize/RsqrtRsqrt+spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:Ъ
spectral_normalize/l2_normalizeMul#spectral_normalize/MatMul:product:0)spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes

:HЁ
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes

:0ѓ
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:0x
'spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       к
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
 *╠╝ї+├
)spectral_normalize/l2_normalize_1/MaximumMaximum.spectral_normalize/l2_normalize_1/Sum:output:04spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:ѕ
'spectral_normalize/l2_normalize_1/RsqrtRsqrt-spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:Ц
!spectral_normalize/l2_normalize_1Mul%spectral_normalize/MatMul_1:product:0+spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

:0
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:0
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes

:Hї
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes

:0▓
spectral_normalize/MatMul_3MatMul%spectral_normalize/MatMul_2:product:0(spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(Э
#spectral_normalize/AssignVariableOpAssignVariableOp1spectral_normalize_matmul_readvariableop_resource(spectral_normalize/StopGradient:output:0)^spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ѕ
!spectral_normalize/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:0*
dtype0е
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:0y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         0   А
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:0ш
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(N
conv2d_transpose_6/ShapeShapeinputs*
T0*
_output_shapes
:p
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
valueB:░
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
value	B :У
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
valueB:И
"conv2d_transpose_6/strided_slice_1StridedSlice!conv2d_transpose_6/stack:output:01conv2d_transpose_6/strided_slice_1/stack:output:03conv2d_transpose_6/strided_slice_1/stack_1:output:03conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask┬
2conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
:0*
dtype0Ѓ
#conv2d_transpose_6/conv2d_transposeConv2DBackpropInput!conv2d_transpose_6/stack:output:0:conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ў
)conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0└
conv2d_transpose_6/BiasAddBiasAdd,conv2d_transpose_6/conv2d_transpose:output:01conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         z
IdentityIdentity#conv2d_transpose_6/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         П
NoOpNoOp^Reshape/ReadVariableOp*^conv2d_transpose_6/BiasAdd/ReadVariableOp3^conv2d_transpose_6/conv2d_transpose/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         	0: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2V
)conv2d_transpose_6/BiasAdd/ReadVariableOp)conv2d_transpose_6/BiasAdd/ReadVariableOp2h
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:W S
/
_output_shapes
:         	0
 
_user_specified_nameinputs"х	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*┼
serving_default▒
<
input_41
serving_default_input_4:0         ђU
spectral_normalization_158
StatefulPartitionedCall:0         tensorflow/serving/predict:Мђ
я
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
Ц
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Н
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
Ж
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
Ц
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses"
_tf_keras_layer
Н
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
Ж
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
Ц
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
Н
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
Ж
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
Ц
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_layer
П
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
┘
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses
	layer
ђw
Ђw_shape
	ѓsn_u
ѓu"
_tf_keras_layer
Џ
$0
Ѓ1
&2
.3
/4
05
16
?7
ё8
A9
I10
J11
K12
L13
Z14
Ё15
\16
e17
f18
g19
h20
є21
Є22
ѕ23
Ѕ24
і25
І26
ї27
ђ28
Ї29
ѓ30"
trackable_list_wrapper
╩
$0
Ѓ1
.2
/3
?4
ё5
I6
J7
Z8
Ё9
e10
f11
є12
Є13
ѕ14
Ѕ15
і16
І17
ї18
ђ19
Ї20"
trackable_list_wrapper
 "
trackable_list_wrapper
¤
јnon_trainable_variables
Јlayers
љmetrics
 Љlayer_regularization_losses
њlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
П
Њtrace_0
ћtrace_1
Ћtrace_2
ќtrace_32Ж
(__inference_model_1_layer_call_fn_387318
(__inference_model_1_layer_call_fn_388189
(__inference_model_1_layer_call_fn_388256
(__inference_model_1_layer_call_fn_387911┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЊtrace_0zћtrace_1zЋtrace_2zќtrace_3
╔
Ќtrace_0
ўtrace_1
Ўtrace_2
џtrace_32о
C__inference_model_1_layer_call_and_return_conditional_losses_388457
C__inference_model_1_layer_call_and_return_conditional_losses_388774
C__inference_model_1_layer_call_and_return_conditional_losses_387986
C__inference_model_1_layer_call_and_return_conditional_losses_388069┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЌtrace_0zўtrace_1zЎtrace_2zџtrace_3
╠B╔
!__inference__wrapped_model_386463input_4"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
-
Џserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
юnon_trainable_variables
Юlayers
ъmetrics
 Ъlayer_regularization_losses
аlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
­
Аtrace_02Л
*__inference_reshape_1_layer_call_fn_388779б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zАtrace_0
І
бtrace_02В
E__inference_reshape_1_layer_call_and_return_conditional_losses_388793б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zбtrace_0
6
$0
Ѓ1
&2"
trackable_list_wrapper
/
$0
Ѓ1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Бnon_trainable_variables
цlayers
Цmetrics
 дlayer_regularization_losses
Дlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
ж
еtrace_0
Еtrace_12«
:__inference_spectral_normalization_12_layer_call_fn_388802
:__inference_spectral_normalization_12_layer_call_fn_388813│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zеtrace_0zЕtrace_1
Ъ
фtrace_0
Фtrace_12С
U__inference_spectral_normalization_12_layer_call_and_return_conditional_losses_388836
U__inference_spectral_normalization_12_layer_call_and_return_conditional_losses_388888│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zфtrace_0zФtrace_1
т
г	variables
Гtrainable_variables
«regularization_losses
»	keras_api
░__call__
+▒&call_and_return_all_conditional_losses

$kernel
	Ѓbias
!▓_jit_compiled_convolution_op"
_tf_keras_layer
;:9	`ђ2 spectral_normalization_12/kernel
 "
trackable_list_wrapper
/:-	ђ2spectral_normalization_12/sn_u
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
▓
│non_trainable_variables
┤layers
хmetrics
 Хlayer_regularization_losses
иlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
р
Иtrace_0
╣trace_12д
6__inference_batch_normalization_3_layer_call_fn_388901
6__inference_batch_normalization_3_layer_call_fn_388914│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zИtrace_0z╣trace_1
Ќ
║trace_0
╗trace_12▄
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_388932
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_388950│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z║trace_0z╗trace_1
 "
trackable_list_wrapper
):'`2batch_normalization_3/gamma
(:&`2batch_normalization_3/beta
1:/` (2!batch_normalization_3/moving_mean
5:3` (2%batch_normalization_3/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╝non_trainable_variables
йlayers
Йmetrics
 ┐layer_regularization_losses
└layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
Ь
┴trace_02¤
(__inference_re_lu_3_layer_call_fn_388955б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┴trace_0
Ѕ
┬trace_02Ж
C__inference_re_lu_3_layer_call_and_return_conditional_losses_388960б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┬trace_0
6
?0
ё1
A2"
trackable_list_wrapper
/
?0
ё1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
├non_trainable_variables
─layers
┼metrics
 кlayer_regularization_losses
Кlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
ж
╚trace_0
╔trace_12«
:__inference_spectral_normalization_13_layer_call_fn_388969
:__inference_spectral_normalization_13_layer_call_fn_388980│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╚trace_0z╔trace_1
Ъ
╩trace_0
╦trace_12С
U__inference_spectral_normalization_13_layer_call_and_return_conditional_losses_389003
U__inference_spectral_normalization_13_layer_call_and_return_conditional_losses_389055│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╩trace_0z╦trace_1
т
╠	variables
═trainable_variables
╬regularization_losses
¤	keras_api
л__call__
+Л&call_and_return_all_conditional_losses

?kernel
	ёbias
!м_jit_compiled_convolution_op"
_tf_keras_layer
::80`2 spectral_normalization_13/kernel
 "
trackable_list_wrapper
.:,`2spectral_normalization_13/sn_u
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
▓
Мnon_trainable_variables
нlayers
Нmetrics
 оlayer_regularization_losses
Оlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
р
пtrace_0
┘trace_12д
6__inference_batch_normalization_4_layer_call_fn_389068
6__inference_batch_normalization_4_layer_call_fn_389081│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zпtrace_0z┘trace_1
Ќ
┌trace_0
█trace_12▄
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_389099
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_389117│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┌trace_0z█trace_1
 "
trackable_list_wrapper
):'02batch_normalization_4/gamma
(:&02batch_normalization_4/beta
1:/0 (2!batch_normalization_4/moving_mean
5:30 (2%batch_normalization_4/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
▄non_trainable_variables
Пlayers
яmetrics
 ▀layer_regularization_losses
Яlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
Ь
рtrace_02¤
(__inference_re_lu_4_layer_call_fn_389122б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zрtrace_0
Ѕ
Рtrace_02Ж
C__inference_re_lu_4_layer_call_and_return_conditional_losses_389127б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zРtrace_0
6
Z0
Ё1
\2"
trackable_list_wrapper
/
Z0
Ё1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
сnon_trainable_variables
Сlayers
тmetrics
 Тlayer_regularization_losses
уlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
ж
Уtrace_0
жtrace_12«
:__inference_spectral_normalization_14_layer_call_fn_389136
:__inference_spectral_normalization_14_layer_call_fn_389147│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zУtrace_0zжtrace_1
Ъ
Жtrace_0
вtrace_12С
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_389170
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_389222│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЖtrace_0zвtrace_1
т
В	variables
ьtrainable_variables
Ьregularization_losses
№	keras_api
­__call__
+ы&call_and_return_all_conditional_losses

Zkernel
	Ёbias
!Ы_jit_compiled_convolution_op"
_tf_keras_layer
::802 spectral_normalization_14/kernel
 "
trackable_list_wrapper
.:,02spectral_normalization_14/sn_u
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
▓
зnon_trainable_variables
Зlayers
шmetrics
 Шlayer_regularization_losses
эlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
р
Эtrace_0
щtrace_12д
6__inference_batch_normalization_5_layer_call_fn_389235
6__inference_batch_normalization_5_layer_call_fn_389248│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЭtrace_0zщtrace_1
Ќ
Щtrace_0
чtrace_12▄
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_389266
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_389284│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЩtrace_0zчtrace_1
 "
trackable_list_wrapper
):'2batch_normalization_5/gamma
(:&2batch_normalization_5/beta
1:/ (2!batch_normalization_5/moving_mean
5:3 (2%batch_normalization_5/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Чnon_trainable_variables
§layers
■metrics
  layer_regularization_losses
ђlayer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
Ь
Ђtrace_02¤
(__inference_re_lu_5_layer_call_fn_389289б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЂtrace_0
Ѕ
ѓtrace_02Ж
C__inference_re_lu_5_layer_call_and_return_conditional_losses_389294б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zѓtrace_0
X
є0
Є1
ѕ2
Ѕ3
і4
І5
ї6"
trackable_list_wrapper
X
є0
Є1
ѕ2
Ѕ3
і4
І5
ї6"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Ѓnon_trainable_variables
ёlayers
Ёmetrics
 єlayer_regularization_losses
Єlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
┘
ѕtrace_0
Ѕtrace_12ъ
2__inference_self_attn_model_1_layer_call_fn_389315
2__inference_self_attn_model_1_layer_call_fn_389336│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zѕtrace_0zЅtrace_1
Ј
іtrace_0
Іtrace_12н
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_389402
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_389468│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zіtrace_0zІtrace_1
п
ї	variables
Їtrainable_variables
јregularization_losses
Ј	keras_api
љ__call__
+Љ&call_and_return_all_conditional_losses
єprivate__attention_1_gamma

єgamma"
_tf_keras_layer
Т
њ	variables
Њtrainable_variables
ћregularization_losses
Ћ	keras_api
ќ__call__
+Ќ&call_and_return_all_conditional_losses
Єkernel
	ѕbias
!ў_jit_compiled_convolution_op"
_tf_keras_layer
Т
Ў	variables
џtrainable_variables
Џregularization_losses
ю	keras_api
Ю__call__
+ъ&call_and_return_all_conditional_losses
Ѕkernel
	іbias
!Ъ_jit_compiled_convolution_op"
_tf_keras_layer
Т
а	variables
Аtrainable_variables
бregularization_losses
Б	keras_api
ц__call__
+Ц&call_and_return_all_conditional_losses
Іkernel
	їbias
!д_jit_compiled_convolution_op"
_tf_keras_layer
8
ђ0
Ї1
ѓ2"
trackable_list_wrapper
0
ђ0
Ї1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Дnon_trainable_variables
еlayers
Еmetrics
 фlayer_regularization_losses
Фlayer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
ж
гtrace_0
Гtrace_12«
:__inference_spectral_normalization_15_layer_call_fn_389477
:__inference_spectral_normalization_15_layer_call_fn_389488│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zгtrace_0zГtrace_1
Ъ
«trace_0
»trace_12С
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_389512
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_389565│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z«trace_0z»trace_1
Т
░	variables
▒trainable_variables
▓regularization_losses
│	keras_api
┤__call__
+х&call_and_return_all_conditional_losses
ђkernel
	Їbias
!Х_jit_compiled_convolution_op"
_tf_keras_layer
::82 spectral_normalization_15/kernel
 "
trackable_list_wrapper
.:,2spectral_normalization_15/sn_u
,:*`2spectral_normalization_12/bias
,:*02spectral_normalization_13/bias
,:*2spectral_normalization_14/bias
K:I 2Aself_attn_model_1/private__attention_1/private__attention_1_gamma
;:92!self_attn_model_1/conv2d_9/kernel
-:+2self_attn_model_1/conv2d_9/bias
<::2"self_attn_model_1/conv2d_10/kernel
.:,2 self_attn_model_1/conv2d_10/bias
<::2"self_attn_model_1/conv2d_11/kernel
.:,2 self_attn_model_1/conv2d_11/bias
,:*2spectral_normalization_15/bias
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
ѓ9"
trackable_list_wrapper
є
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
ЩBэ
(__inference_model_1_layer_call_fn_387318input_4"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
(__inference_model_1_layer_call_fn_388189inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
(__inference_model_1_layer_call_fn_388256inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
(__inference_model_1_layer_call_fn_387911input_4"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћBЉ
C__inference_model_1_layer_call_and_return_conditional_losses_388457inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћBЉ
C__inference_model_1_layer_call_and_return_conditional_losses_388774inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЋBњ
C__inference_model_1_layer_call_and_return_conditional_losses_387986input_4"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЋBњ
C__inference_model_1_layer_call_and_return_conditional_losses_388069input_4"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╦B╚
$__inference_signature_wrapper_388130input_4"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
яB█
*__inference_reshape_1_layer_call_fn_388779inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
E__inference_reshape_1_layer_call_and_return_conditional_losses_388793inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
 BЧ
:__inference_spectral_normalization_12_layer_call_fn_388802inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
:__inference_spectral_normalization_12_layer_call_fn_388813inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џBЌ
U__inference_spectral_normalization_12_layer_call_and_return_conditional_losses_388836inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џBЌ
U__inference_spectral_normalization_12_layer_call_and_return_conditional_losses_388888inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
/
$0
Ѓ1"
trackable_list_wrapper
/
$0
Ѓ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
иnon_trainable_variables
Иlayers
╣metrics
 ║layer_regularization_losses
╗layer_metrics
г	variables
Гtrainable_variables
«regularization_losses
░__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses"
_generic_user_object
щ
╝trace_02┌
3__inference_conv2d_transpose_4_layer_call_fn_389574б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╝trace_0
ћ
йtrace_02ш
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_389607б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zйtrace_0
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
чBЭ
6__inference_batch_normalization_3_layer_call_fn_388901inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
чBЭ
6__inference_batch_normalization_3_layer_call_fn_388914inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ќBЊ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_388932inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ќBЊ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_388950inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
▄B┘
(__inference_re_lu_3_layer_call_fn_388955inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
эBЗ
C__inference_re_lu_3_layer_call_and_return_conditional_losses_388960inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
 BЧ
:__inference_spectral_normalization_13_layer_call_fn_388969inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
:__inference_spectral_normalization_13_layer_call_fn_388980inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џBЌ
U__inference_spectral_normalization_13_layer_call_and_return_conditional_losses_389003inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џBЌ
U__inference_spectral_normalization_13_layer_call_and_return_conditional_losses_389055inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
/
?0
ё1"
trackable_list_wrapper
/
?0
ё1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Йnon_trainable_variables
┐layers
└metrics
 ┴layer_regularization_losses
┬layer_metrics
╠	variables
═trainable_variables
╬regularization_losses
л__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
щ
├trace_02┌
3__inference_conv2d_transpose_5_layer_call_fn_389616б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z├trace_0
ћ
─trace_02ш
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_389649б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z─trace_0
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
чBЭ
6__inference_batch_normalization_4_layer_call_fn_389068inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
чBЭ
6__inference_batch_normalization_4_layer_call_fn_389081inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ќBЊ
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_389099inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ќBЊ
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_389117inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
▄B┘
(__inference_re_lu_4_layer_call_fn_389122inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
эBЗ
C__inference_re_lu_4_layer_call_and_return_conditional_losses_389127inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
 BЧ
:__inference_spectral_normalization_14_layer_call_fn_389136inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
:__inference_spectral_normalization_14_layer_call_fn_389147inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џBЌ
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_389170inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џBЌ
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_389222inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
/
Z0
Ё1"
trackable_list_wrapper
/
Z0
Ё1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
┼non_trainable_variables
кlayers
Кmetrics
 ╚layer_regularization_losses
╔layer_metrics
В	variables
ьtrainable_variables
Ьregularization_losses
­__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
щ
╩trace_02┌
3__inference_conv2d_transpose_6_layer_call_fn_389658б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╩trace_0
ћ
╦trace_02ш
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_389691б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╦trace_0
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
чBЭ
6__inference_batch_normalization_5_layer_call_fn_389235inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
чBЭ
6__inference_batch_normalization_5_layer_call_fn_389248inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ќBЊ
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_389266inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ќBЊ
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_389284inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
▄B┘
(__inference_re_lu_5_layer_call_fn_389289inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
эBЗ
C__inference_re_lu_5_layer_call_and_return_conditional_losses_389294inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
эBЗ
2__inference_self_attn_model_1_layer_call_fn_389315inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
эBЗ
2__inference_self_attn_model_1_layer_call_fn_389336inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
њBЈ
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_389402inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
њBЈ
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_389468inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
(
є0"
trackable_list_wrapper
(
є0"
trackable_list_wrapper
 "
trackable_list_wrapper
И
╠non_trainable_variables
═layers
╬metrics
 ¤layer_regularization_losses
лlayer_metrics
ї	variables
Їtrainable_variables
јregularization_losses
љ__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
ч
Лtrace_02▄
5__inference_private__attention_1_layer_call_fn_389703б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЛtrace_0
ќ
мtrace_02э
P__inference_private__attention_1_layer_call_and_return_conditional_losses_389754б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zмtrace_0
0
Є0
ѕ1"
trackable_list_wrapper
0
Є0
ѕ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Мnon_trainable_variables
нlayers
Нmetrics
 оlayer_regularization_losses
Оlayer_metrics
њ	variables
Њtrainable_variables
ћregularization_losses
ќ__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses"
_generic_user_object
№
пtrace_02л
)__inference_conv2d_9_layer_call_fn_389763б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zпtrace_0
і
┘trace_02в
D__inference_conv2d_9_layer_call_and_return_conditional_losses_389773б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┘trace_0
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
0
Ѕ0
і1"
trackable_list_wrapper
0
Ѕ0
і1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
┌non_trainable_variables
█layers
▄metrics
 Пlayer_regularization_losses
яlayer_metrics
Ў	variables
џtrainable_variables
Џregularization_losses
Ю__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
­
▀trace_02Л
*__inference_conv2d_10_layer_call_fn_389782б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z▀trace_0
І
Яtrace_02В
E__inference_conv2d_10_layer_call_and_return_conditional_losses_389792б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЯtrace_0
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
0
І0
ї1"
trackable_list_wrapper
0
І0
ї1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
рnon_trainable_variables
Рlayers
сmetrics
 Сlayer_regularization_losses
тlayer_metrics
а	variables
Аtrainable_variables
бregularization_losses
ц__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
­
Тtrace_02Л
*__inference_conv2d_11_layer_call_fn_389801б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zТtrace_0
І
уtrace_02В
E__inference_conv2d_11_layer_call_and_return_conditional_losses_389811б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zуtrace_0
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
(
ѓ0"
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
 BЧ
:__inference_spectral_normalization_15_layer_call_fn_389477inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
:__inference_spectral_normalization_15_layer_call_fn_389488inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џBЌ
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_389512inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џBЌ
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_389565inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
0
ђ0
Ї1"
trackable_list_wrapper
0
ђ0
Ї1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Уnon_trainable_variables
жlayers
Жmetrics
 вlayer_regularization_losses
Вlayer_metrics
░	variables
▒trainable_variables
▓regularization_losses
┤__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
щ
ьtrace_02┌
3__inference_conv2d_transpose_7_layer_call_fn_389820б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zьtrace_0
ћ
Ьtrace_02ш
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_389854б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЬtrace_0
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
уBС
3__inference_conv2d_transpose_4_layer_call_fn_389574inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ѓB 
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_389607inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
уBС
3__inference_conv2d_transpose_5_layer_call_fn_389616inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ѓB 
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_389649inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
уBС
3__inference_conv2d_transpose_6_layer_call_fn_389658inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ѓB 
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_389691inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
ЅBє
5__inference_private__attention_1_layer_call_fn_389703inputs/0inputs/1inputs/2inputs/3"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
цBА
P__inference_private__attention_1_layer_call_and_return_conditional_losses_389754inputs/0inputs/1inputs/2inputs/3"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
ПB┌
)__inference_conv2d_9_layer_call_fn_389763inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
D__inference_conv2d_9_layer_call_and_return_conditional_losses_389773inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
яB█
*__inference_conv2d_10_layer_call_fn_389782inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
E__inference_conv2d_10_layer_call_and_return_conditional_losses_389792inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
яB█
*__inference_conv2d_11_layer_call_fn_389801inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
E__inference_conv2d_11_layer_call_and_return_conditional_losses_389811inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
уBС
3__inference_conv2d_transpose_7_layer_call_fn_389820inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ѓB 
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_389854inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 р
!__inference__wrapped_model_386463╗'$Ѓ./01?ёIJKLZЁefghЄѕЅіІїєђЇ1б.
'б$
"і
input_4         ђ
ф "]фZ
X
spectral_normalization_15;і8
spectral_normalization_15         В
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_388932ќ./01MбJ
Cб@
:і7
inputs+                           `
p 
ф "?б<
5і2
0+                           `
џ В
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_388950ќ./01MбJ
Cб@
:і7
inputs+                           `
p
ф "?б<
5і2
0+                           `
џ ─
6__inference_batch_normalization_3_layer_call_fn_388901Ѕ./01MбJ
Cб@
:і7
inputs+                           `
p 
ф "2і/+                           `─
6__inference_batch_normalization_3_layer_call_fn_388914Ѕ./01MбJ
Cб@
:і7
inputs+                           `
p
ф "2і/+                           `В
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_389099ќIJKLMбJ
Cб@
:і7
inputs+                           0
p 
ф "?б<
5і2
0+                           0
џ В
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_389117ќIJKLMбJ
Cб@
:і7
inputs+                           0
p
ф "?б<
5і2
0+                           0
џ ─
6__inference_batch_normalization_4_layer_call_fn_389068ЅIJKLMбJ
Cб@
:і7
inputs+                           0
p 
ф "2і/+                           0─
6__inference_batch_normalization_4_layer_call_fn_389081ЅIJKLMбJ
Cб@
:і7
inputs+                           0
p
ф "2і/+                           0В
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_389266ќefghMбJ
Cб@
:і7
inputs+                           
p 
ф "?б<
5і2
0+                           
џ В
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_389284ќefghMбJ
Cб@
:і7
inputs+                           
p
ф "?б<
5і2
0+                           
џ ─
6__inference_batch_normalization_5_layer_call_fn_389235ЅefghMбJ
Cб@
:і7
inputs+                           
p 
ф "2і/+                           ─
6__inference_batch_normalization_5_layer_call_fn_389248ЅefghMбJ
Cб@
:і7
inputs+                           
p
ф "2і/+                           и
E__inference_conv2d_10_layer_call_and_return_conditional_losses_389792nЅі7б4
-б*
(і%
inputs         
ф "-б*
#і 
0         
џ Ј
*__inference_conv2d_10_layer_call_fn_389782aЅі7б4
-б*
(і%
inputs         
ф " і         и
E__inference_conv2d_11_layer_call_and_return_conditional_losses_389811nІї7б4
-б*
(і%
inputs         
ф "-б*
#і 
0         
џ Ј
*__inference_conv2d_11_layer_call_fn_389801aІї7б4
-б*
(і%
inputs         
ф " і         Х
D__inference_conv2d_9_layer_call_and_return_conditional_losses_389773nЄѕ7б4
-б*
(і%
inputs         
ф "-б*
#і 
0         
џ ј
)__inference_conv2d_9_layer_call_fn_389763aЄѕ7б4
-б*
(і%
inputs         
ф " і         т
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_389607њ$ЃJбG
@б=
;і8
inputs,                           ђ
ф "?б<
5і2
0+                           `
џ й
3__inference_conv2d_transpose_4_layer_call_fn_389574Ё$ЃJбG
@б=
;і8
inputs,                           ђ
ф "2і/+                           `С
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_389649Љ?ёIбF
?б<
:і7
inputs+                           `
ф "?б<
5і2
0+                           0
џ ╝
3__inference_conv2d_transpose_5_layer_call_fn_389616ё?ёIбF
?б<
:і7
inputs+                           `
ф "2і/+                           0С
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_389691ЉZЁIбF
?б<
:і7
inputs+                           0
ф "?б<
5і2
0+                           
џ ╝
3__inference_conv2d_transpose_6_layer_call_fn_389658ёZЁIбF
?б<
:і7
inputs+                           0
ф "2і/+                           т
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_389854њђЇIбF
?б<
:і7
inputs+                           
ф "?б<
5і2
0+                           
џ й
3__inference_conv2d_transpose_7_layer_call_fn_389820ЁђЇIбF
?б<
:і7
inputs+                           
ф "2і/+                           █
C__inference_model_1_layer_call_and_return_conditional_losses_387986Њ'$Ѓ./01?ёIJKLZЁefghЄѕЅіІїєђЇ9б6
/б,
"і
input_4         ђ
p 

 
ф "-б*
#і 
0         
џ Я
C__inference_model_1_layer_call_and_return_conditional_losses_388069ў,$&Ѓ./01?AёIJKLZ\ЁefghЄѕЅіІїєђѓЇ9б6
/б,
"і
input_4         ђ
p

 
ф "-б*
#і 
0         
џ ┌
C__inference_model_1_layer_call_and_return_conditional_losses_388457њ'$Ѓ./01?ёIJKLZЁefghЄѕЅіІїєђЇ8б5
.б+
!і
inputs         ђ
p 

 
ф "-б*
#і 
0         
џ ▀
C__inference_model_1_layer_call_and_return_conditional_losses_388774Ќ,$&Ѓ./01?AёIJKLZ\ЁefghЄѕЅіІїєђѓЇ8б5
.б+
!і
inputs         ђ
p

 
ф "-б*
#і 
0         
џ │
(__inference_model_1_layer_call_fn_387318є'$Ѓ./01?ёIJKLZЁefghЄѕЅіІїєђЇ9б6
/б,
"і
input_4         ђ
p 

 
ф " і         И
(__inference_model_1_layer_call_fn_387911І,$&Ѓ./01?AёIJKLZ\ЁefghЄѕЅіІїєђѓЇ9б6
/б,
"і
input_4         ђ
p

 
ф " і         ▓
(__inference_model_1_layer_call_fn_388189Ё'$Ѓ./01?ёIJKLZЁefghЄѕЅіІїєђЇ8б5
.б+
!і
inputs         ђ
p 

 
ф " і         и
(__inference_model_1_layer_call_fn_388256і,$&Ѓ./01?AёIJKLZ\ЁefghЄѕЅіІїєђѓЇ8б5
.б+
!і
inputs         ђ
p

 
ф " і         Ч
P__inference_private__attention_1_layer_call_and_return_conditional_losses_389754ДєКб├
╗би
┤џ░
*і'
inputs/0         
*і'
inputs/1         
*і'
inputs/2         
*і'
inputs/3         
ф "WбT
MбJ
%і"
0/0         
!і
0/1         DD
џ М
5__inference_private__attention_1_layer_call_fn_389703ЎєКб├
╗би
┤џ░
*і'
inputs/0         
*і'
inputs/1         
*і'
inputs/2         
*і'
inputs/3         
ф "IбF
#і 
0         
і
1         DD»
C__inference_re_lu_3_layer_call_and_return_conditional_losses_388960h7б4
-б*
(і%
inputs         `
ф "-б*
#і 
0         `
џ Є
(__inference_re_lu_3_layer_call_fn_388955[7б4
-б*
(і%
inputs         `
ф " і         `»
C__inference_re_lu_4_layer_call_and_return_conditional_losses_389127h7б4
-б*
(і%
inputs         	0
ф "-б*
#і 
0         	0
џ Є
(__inference_re_lu_4_layer_call_fn_389122[7б4
-б*
(і%
inputs         	0
ф " і         	0»
C__inference_re_lu_5_layer_call_and_return_conditional_losses_389294h7б4
-б*
(і%
inputs         
ф "-б*
#і 
0         
џ Є
(__inference_re_lu_5_layer_call_fn_389289[7б4
-б*
(і%
inputs         
ф " і         Ф
E__inference_reshape_1_layer_call_and_return_conditional_losses_388793b0б-
&б#
!і
inputs         ђ
ф ".б+
$і!
0         ђ
џ Ѓ
*__inference_reshape_1_layer_call_fn_388779U0б-
&б#
!і
inputs         ђ
ф "!і         ђЭ
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_389402дЄѕЅіІїє;б8
1б.
(і%
inputs         
p 
ф "WбT
MбJ
%і"
0/0         
!і
0/1         DD
џ Э
M__inference_self_attn_model_1_layer_call_and_return_conditional_losses_389468дЄѕЅіІїє;б8
1б.
(і%
inputs         
p
ф "WбT
MбJ
%і"
0/0         
!і
0/1         DD
џ ¤
2__inference_self_attn_model_1_layer_call_fn_389315ўЄѕЅіІїє;б8
1б.
(і%
inputs         
p 
ф "IбF
#і 
0         
і
1         DD¤
2__inference_self_attn_model_1_layer_call_fn_389336ўЄѕЅіІїє;б8
1б.
(і%
inputs         
p
ф "IбF
#і 
0         
і
1         DD№
$__inference_signature_wrapper_388130к'$Ѓ./01?ёIJKLZЁefghЄѕЅіІїєђЇ<б9
б 
2ф/
-
input_4"і
input_4         ђ"]фZ
X
spectral_normalization_15;і8
spectral_normalization_15         ╦
U__inference_spectral_normalization_12_layer_call_and_return_conditional_losses_388836r$Ѓ<б9
2б/
)і&
inputs         ђ
p 
ф "-б*
#і 
0         `
џ ╠
U__inference_spectral_normalization_12_layer_call_and_return_conditional_losses_388888s$&Ѓ<б9
2б/
)і&
inputs         ђ
p
ф "-б*
#і 
0         `
џ Б
:__inference_spectral_normalization_12_layer_call_fn_388802e$Ѓ<б9
2б/
)і&
inputs         ђ
p 
ф " і         `ц
:__inference_spectral_normalization_12_layer_call_fn_388813f$&Ѓ<б9
2б/
)і&
inputs         ђ
p
ф " і         `╩
U__inference_spectral_normalization_13_layer_call_and_return_conditional_losses_389003q?ё;б8
1б.
(і%
inputs         `
p 
ф "-б*
#і 
0         	0
џ ╦
U__inference_spectral_normalization_13_layer_call_and_return_conditional_losses_389055r?Aё;б8
1б.
(і%
inputs         `
p
ф "-б*
#і 
0         	0
џ б
:__inference_spectral_normalization_13_layer_call_fn_388969d?ё;б8
1б.
(і%
inputs         `
p 
ф " і         	0Б
:__inference_spectral_normalization_13_layer_call_fn_388980e?Aё;б8
1б.
(і%
inputs         `
p
ф " і         	0╩
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_389170qZЁ;б8
1б.
(і%
inputs         	0
p 
ф "-б*
#і 
0         
џ ╦
U__inference_spectral_normalization_14_layer_call_and_return_conditional_losses_389222rZ\Ё;б8
1б.
(і%
inputs         	0
p
ф "-б*
#і 
0         
џ б
:__inference_spectral_normalization_14_layer_call_fn_389136dZЁ;б8
1б.
(і%
inputs         	0
p 
ф " і         Б
:__inference_spectral_normalization_14_layer_call_fn_389147eZ\Ё;б8
1б.
(і%
inputs         	0
p
ф " і         ╦
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_389512rђЇ;б8
1б.
(і%
inputs         
p 
ф "-б*
#і 
0         
џ ═
U__inference_spectral_normalization_15_layer_call_and_return_conditional_losses_389565tђѓЇ;б8
1б.
(і%
inputs         
p
ф "-б*
#і 
0         
џ Б
:__inference_spectral_normalization_15_layer_call_fn_389477eђЇ;б8
1б.
(і%
inputs         
p 
ф " і         Ц
:__inference_spectral_normalization_15_layer_call_fn_389488gђѓЇ;б8
1б.
(і%
inputs         
p
ф " і         
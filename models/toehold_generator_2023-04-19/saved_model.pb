юИ
ЎЧ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
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
Ў
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
n
	ReverseV2
tensor"T
axis"Tidx
output"T"
Tidxtype0:
2	"
Ttype:
2	

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
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.10.02unknown8пЧ
ћ
spectral_normalization_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name spectral_normalization_19/bias
Ї
2spectral_normalization_19/bias/Read/ReadVariableOpReadVariableOpspectral_normalization_19/bias*
_output_shapes
:*
dtype0
ћ
spectral_normalization_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name spectral_normalization_18/bias
Ї
2spectral_normalization_18/bias/Read/ReadVariableOpReadVariableOpspectral_normalization_18/bias*
_output_shapes
:*
dtype0
ћ
spectral_normalization_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name spectral_normalization_17/bias
Ї
2spectral_normalization_17/bias/Read/ReadVariableOpReadVariableOpspectral_normalization_17/bias*
_output_shapes
: *
dtype0
ћ
spectral_normalization_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name spectral_normalization_16/bias
Ї
2spectral_normalization_16/bias/Read/ReadVariableOpReadVariableOpspectral_normalization_16/bias*
_output_shapes
:@*
dtype0
ц
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_11/moving_variance
Ю
:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes
:*
dtype0
ю
"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_11/moving_mean
Ћ
6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes
:*
dtype0
ј
batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_11/beta
Є
/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes
:*
dtype0
љ
batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_11/gamma
Ѕ
0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*
_output_shapes
:*
dtype0
ў
spectral_normalization_19/sn_uVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name spectral_normalization_19/sn_u
Љ
2spectral_normalization_19/sn_u/Read/ReadVariableOpReadVariableOpspectral_normalization_19/sn_u*
_output_shapes

:*
dtype0
ц
 spectral_normalization_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" spectral_normalization_19/kernel
Ю
4spectral_normalization_19/kernel/Read/ReadVariableOpReadVariableOp spectral_normalization_19/kernel*&
_output_shapes
:*
dtype0
ц
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_10/moving_variance
Ю
:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes
:*
dtype0
ю
"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_10/moving_mean
Ћ
6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes
:*
dtype0
ј
batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_10/beta
Є
/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes
:*
dtype0
љ
batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_10/gamma
Ѕ
0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*
_output_shapes
:*
dtype0
ў
spectral_normalization_18/sn_uVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name spectral_normalization_18/sn_u
Љ
2spectral_normalization_18/sn_u/Read/ReadVariableOpReadVariableOpspectral_normalization_18/sn_u*
_output_shapes

:*
dtype0
ц
 spectral_normalization_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" spectral_normalization_18/kernel
Ю
4spectral_normalization_18/kernel/Read/ReadVariableOpReadVariableOp spectral_normalization_18/kernel*&
_output_shapes
: *
dtype0
б
%batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_9/moving_variance
Џ
9batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*
_output_shapes
: *
dtype0
џ
!batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_9/moving_mean
Њ
5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
_output_shapes
: *
dtype0
ї
batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_9/beta
Ё
.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
_output_shapes
: *
dtype0
ј
batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_9/gamma
Є
/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
_output_shapes
: *
dtype0
ў
spectral_normalization_17/sn_uVarHandleOp*
_output_shapes
: *
dtype0*
shape
: */
shared_name spectral_normalization_17/sn_u
Љ
2spectral_normalization_17/sn_u/Read/ReadVariableOpReadVariableOpspectral_normalization_17/sn_u*
_output_shapes

: *
dtype0
ц
 spectral_normalization_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *1
shared_name" spectral_normalization_17/kernel
Ю
4spectral_normalization_17/kernel/Read/ReadVariableOpReadVariableOp spectral_normalization_17/kernel*&
_output_shapes
:@ *
dtype0
б
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_8/moving_variance
Џ
9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes
:@*
dtype0
џ
!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_8/moving_mean
Њ
5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes
:@*
dtype0
ї
batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_8/beta
Ё
.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes
:@*
dtype0
ј
batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_8/gamma
Є
/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes
:@*
dtype0
ў
spectral_normalization_16/sn_uVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*/
shared_name spectral_normalization_16/sn_u
Љ
2spectral_normalization_16/sn_u/Read/ReadVariableOpReadVariableOpspectral_normalization_16/sn_u*
_output_shapes

:@*
dtype0
Ц
 spectral_normalization_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ@*1
shared_name" spectral_normalization_16/kernel
ъ
4spectral_normalization_16/kernel/Read/ReadVariableOpReadVariableOp spectral_normalization_16/kernel*'
_output_shapes
:	ђ@*
dtype0
|
serving_default_input_5Placeholder*(
_output_shapes
:         ђ*
dtype0*
shape:         ђ
і	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5 spectral_normalization_16/kernelspectral_normalization_16/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variance spectral_normalization_17/kernelspectral_normalization_17/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_variance spectral_normalization_18/kernelspectral_normalization_18/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_variance spectral_normalization_19/kernelspectral_normalization_19/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_variance*$
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         <*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *.
f)R'
%__inference_signature_wrapper_2462072

NoOpNoOp
┤t
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*№s
valueтsBРs B█s
Е
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer_with_weights-6
layer-19
layer_with_weights-7
layer-20
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
ј
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses* 
ј
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses* 
└
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
	0layer
1w
2w_shape
3sn_u
3u*
Н
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses
:axis
	;gamma
<beta
=moving_mean
>moving_variance*
ј
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses* 
└
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses
	Klayer
Lw
Mw_shape
Nsn_u
Nu*
Н
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
Uaxis
	Vgamma
Wbeta
Xmoving_mean
Ymoving_variance*
ј
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses* 
└
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses
	flayer
gw
hw_shape
isn_u
iu*
Н
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses
paxis
	qgamma
rbeta
smoving_mean
tmoving_variance*

u	keras_api* 

v	keras_api* 

w	keras_api* 

x	keras_api* 

y	keras_api* 

z	keras_api* 

{	keras_api* 

|	keras_api* 
╚
}	variables
~trainable_variables
regularization_losses
ђ	keras_api
Ђ__call__
+ѓ&call_and_return_all_conditional_losses

Ѓlayer
ёw
Ёw_shape
	єsn_u
єu*
Я
Є	variables
ѕtrainable_variables
Ѕregularization_losses
і	keras_api
І__call__
+ї&call_and_return_all_conditional_losses
	Їaxis

јgamma
	Јbeta
љmoving_mean
Љmoving_variance*
С
10
њ1
32
;3
<4
=5
>6
L7
Њ8
N9
V10
W11
X12
Y13
g14
ћ15
i16
q17
r18
s19
t20
ё21
Ћ22
є23
ј24
Ј25
љ26
Љ27*
Ђ
10
њ1
;2
<3
L4
Њ5
V6
W7
g8
ћ9
q10
r11
ё12
Ћ13
ј14
Ј15*
* 
х
ќnon_trainable_variables
Ќlayers
ўmetrics
 Ўlayer_regularization_losses
џlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
Џtrace_0
юtrace_1
Юtrace_2
ъtrace_3* 
:
Ъtrace_0
аtrace_1
Аtrace_2
бtrace_3* 
* 

Бserving_default* 
* 
* 
* 
ќ
цnon_trainable_variables
Цlayers
дmetrics
 Дlayer_regularization_losses
еlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses* 

Еtrace_0* 

фtrace_0* 
* 
* 
* 
ќ
Фnon_trainable_variables
гlayers
Гmetrics
 «layer_regularization_losses
»layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses* 

░trace_0* 

▒trace_0* 

10
њ1
32*

10
њ1*
* 
ў
▓non_trainable_variables
│layers
┤metrics
 хlayer_regularization_losses
Хlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*

иtrace_0
Иtrace_1* 

╣trace_0
║trace_1* 
л
╗	variables
╝trainable_variables
йregularization_losses
Й	keras_api
┐__call__
+└&call_and_return_all_conditional_losses

1kernel
	њbias
!┴_jit_compiled_convolution_op*
ke
VARIABLE_VALUE spectral_normalization_16/kernel1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUE*
* 
lf
VARIABLE_VALUEspectral_normalization_16/sn_u4layer_with_weights-0/sn_u/.ATTRIBUTES/VARIABLE_VALUE*
 
;0
<1
=2
>3*

;0
<1*
* 
ў
┬non_trainable_variables
├layers
─metrics
 ┼layer_regularization_losses
кlayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*

Кtrace_0
╚trace_1* 

╔trace_0
╩trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_8/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_8/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_8/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_8/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
ќ
╦non_trainable_variables
╠layers
═metrics
 ╬layer_regularization_losses
¤layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses* 

лtrace_0* 

Лtrace_0* 

L0
Њ1
N2*

L0
Њ1*
* 
ў
мnon_trainable_variables
Мlayers
нmetrics
 Нlayer_regularization_losses
оlayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*

Оtrace_0
пtrace_1* 

┘trace_0
┌trace_1* 
л
█	variables
▄trainable_variables
Пregularization_losses
я	keras_api
▀__call__
+Я&call_and_return_all_conditional_losses

Lkernel
	Њbias
!р_jit_compiled_convolution_op*
ke
VARIABLE_VALUE spectral_normalization_17/kernel1layer_with_weights-2/w/.ATTRIBUTES/VARIABLE_VALUE*
* 
lf
VARIABLE_VALUEspectral_normalization_17/sn_u4layer_with_weights-2/sn_u/.ATTRIBUTES/VARIABLE_VALUE*
 
V0
W1
X2
Y3*

V0
W1*
* 
ў
Рnon_trainable_variables
сlayers
Сmetrics
 тlayer_regularization_losses
Тlayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*

уtrace_0
Уtrace_1* 

жtrace_0
Жtrace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_9/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_9/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_9/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_9/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
ќ
вnon_trainable_variables
Вlayers
ьmetrics
 Ьlayer_regularization_losses
№layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses* 

­trace_0* 

ыtrace_0* 

g0
ћ1
i2*

g0
ћ1*
* 
ў
Ыnon_trainable_variables
зlayers
Зmetrics
 шlayer_regularization_losses
Шlayer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses*

эtrace_0
Эtrace_1* 

щtrace_0
Щtrace_1* 
л
ч	variables
Чtrainable_variables
§regularization_losses
■	keras_api
 __call__
+ђ&call_and_return_all_conditional_losses

gkernel
	ћbias
!Ђ_jit_compiled_convolution_op*
ke
VARIABLE_VALUE spectral_normalization_18/kernel1layer_with_weights-4/w/.ATTRIBUTES/VARIABLE_VALUE*
* 
lf
VARIABLE_VALUEspectral_normalization_18/sn_u4layer_with_weights-4/sn_u/.ATTRIBUTES/VARIABLE_VALUE*
 
q0
r1
s2
t3*

q0
r1*
* 
ў
ѓnon_trainable_variables
Ѓlayers
ёmetrics
 Ёlayer_regularization_losses
єlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses*

Єtrace_0
ѕtrace_1* 

Ѕtrace_0
іtrace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_10/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_10/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_10/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_10/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 

ё0
Ћ1
є2*

ё0
Ћ1*
* 
Џ
Іnon_trainable_variables
їlayers
Їmetrics
 јlayer_regularization_losses
Јlayer_metrics
}	variables
~trainable_variables
regularization_losses
Ђ__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses*

љtrace_0
Љtrace_1* 

њtrace_0
Њtrace_1* 
Л
ћ	variables
Ћtrainable_variables
ќregularization_losses
Ќ	keras_api
ў__call__
+Ў&call_and_return_all_conditional_losses
ёkernel
	Ћbias
!џ_jit_compiled_convolution_op*
ke
VARIABLE_VALUE spectral_normalization_19/kernel1layer_with_weights-6/w/.ATTRIBUTES/VARIABLE_VALUE*
* 
lf
VARIABLE_VALUEspectral_normalization_19/sn_u4layer_with_weights-6/sn_u/.ATTRIBUTES/VARIABLE_VALUE*
$
ј0
Ј1
љ2
Љ3*

ј0
Ј1*
* 
ъ
Џnon_trainable_variables
юlayers
Юmetrics
 ъlayer_regularization_losses
Ъlayer_metrics
Є	variables
ѕtrainable_variables
Ѕregularization_losses
І__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses*

аtrace_0
Аtrace_1* 

бtrace_0
Бtrace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_11/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_11/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_11/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_11/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEspectral_normalization_16/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEspectral_normalization_17/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEspectral_normalization_18/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEspectral_normalization_19/bias'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
]
30
=1
>2
N3
X4
Y5
i6
s7
t8
є9
љ10
Љ11*
б
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
13
14
15
16
17
18
19
20*
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

30*

00*
* 
* 
* 
* 
* 
* 
* 

10
њ1*

10
њ1*
* 
ъ
цnon_trainable_variables
Цlayers
дmetrics
 Дlayer_regularization_losses
еlayer_metrics
╗	variables
╝trainable_variables
йregularization_losses
┐__call__
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses*
* 
* 
* 

=0
>1*
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

N0*

K0*
* 
* 
* 
* 
* 
* 
* 

L0
Њ1*

L0
Њ1*
* 
ъ
Еnon_trainable_variables
фlayers
Фmetrics
 гlayer_regularization_losses
Гlayer_metrics
█	variables
▄trainable_variables
Пregularization_losses
▀__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses*
* 
* 
* 

X0
Y1*
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

i0*

f0*
* 
* 
* 
* 
* 
* 
* 

g0
ћ1*

g0
ћ1*
* 
ъ
«non_trainable_variables
»layers
░metrics
 ▒layer_regularization_losses
▓layer_metrics
ч	variables
Чtrainable_variables
§regularization_losses
 __call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses*
* 
* 
* 

s0
t1*
* 
* 
* 
* 
* 
* 
* 
* 

є0*

Ѓ0*
* 
* 
* 
* 
* 
* 
* 

ё0
Ћ1*

ё0
Ћ1*
* 
ъ
│non_trainable_variables
┤layers
хmetrics
 Хlayer_regularization_losses
иlayer_metrics
ћ	variables
Ћtrainable_variables
ќregularization_losses
ў__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses*
* 
* 
* 

љ0
Љ1*
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
Є
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename4spectral_normalization_16/kernel/Read/ReadVariableOp2spectral_normalization_16/sn_u/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp4spectral_normalization_17/kernel/Read/ReadVariableOp2spectral_normalization_17/sn_u/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOp4spectral_normalization_18/kernel/Read/ReadVariableOp2spectral_normalization_18/sn_u/Read/ReadVariableOp0batch_normalization_10/gamma/Read/ReadVariableOp/batch_normalization_10/beta/Read/ReadVariableOp6batch_normalization_10/moving_mean/Read/ReadVariableOp:batch_normalization_10/moving_variance/Read/ReadVariableOp4spectral_normalization_19/kernel/Read/ReadVariableOp2spectral_normalization_19/sn_u/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOp2spectral_normalization_16/bias/Read/ReadVariableOp2spectral_normalization_17/bias/Read/ReadVariableOp2spectral_normalization_18/bias/Read/ReadVariableOp2spectral_normalization_19/bias/Read/ReadVariableOpConst*)
Tin"
 2*
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
GPU2*0J 8ѓ *)
f$R"
 __inference__traced_save_2463287
м	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename spectral_normalization_16/kernelspectral_normalization_16/sn_ubatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variance spectral_normalization_17/kernelspectral_normalization_17/sn_ubatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_variance spectral_normalization_18/kernelspectral_normalization_18/sn_ubatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_variance spectral_normalization_19/kernelspectral_normalization_19/sn_ubatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_variancespectral_normalization_16/biasspectral_normalization_17/biasspectral_normalization_18/biasspectral_normalization_19/bias*(
Tin!
2*
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
GPU2*0J 8ѓ *,
f'R%
#__inference__traced_restore_2463381ЌБ
П
▒
;__inference_spectral_normalization_16_layer_call_fn_2462623

inputs"
unknown:	ђ@
	unknown_0:@
identityѕбStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *_
fZRX
V__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_2461125Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
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
╬
ъ
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2461048

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╬
ъ
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2463029

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ѕ
┬
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2461079

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
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
-:+                           н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Т
И
V__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_2462645

inputsC
(conv2d_14_conv2d_readvariableop_resource:	ђ@7
)conv2d_14_biasadd_readvariableop_resource:@
identityѕб conv2d_14/BiasAdd/ReadVariableOpбconv2d_14/Conv2D/ReadVariableOpЉ
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*'
_output_shapes
:	ђ@*
dtype0┐
conv2d_14/Conv2DConv2Dinputs'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
є
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Г
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @~
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @Ё
IdentityIdentityconv2d_14/Relu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           @І
NoOpNoOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           ђ: : 2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
ќ	
м
7__inference_batch_normalization_9_layer_call_fn_2462848

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityѕбStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2460901Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Лu
є
D__inference_model_2_layer_call_and_return_conditional_losses_2461249

inputs<
!spectral_normalization_16_2461126:	ђ@/
!spectral_normalization_16_2461128:@+
batch_normalization_8_2461131:@+
batch_normalization_8_2461133:@+
batch_normalization_8_2461135:@+
batch_normalization_8_2461137:@;
!spectral_normalization_17_2461153:@ /
!spectral_normalization_17_2461155: +
batch_normalization_9_2461158: +
batch_normalization_9_2461160: +
batch_normalization_9_2461162: +
batch_normalization_9_2461164: ;
!spectral_normalization_18_2461180: /
!spectral_normalization_18_2461182:,
batch_normalization_10_2461185:,
batch_normalization_10_2461187:,
batch_normalization_10_2461189:,
batch_normalization_10_2461191:;
!spectral_normalization_19_2461234:/
!spectral_normalization_19_2461236:,
batch_normalization_11_2461239:,
batch_normalization_11_2461241:,
batch_normalization_11_2461243:,
batch_normalization_11_2461245:
identityѕб.batch_normalization_10/StatefulPartitionedCallб.batch_normalization_11/StatefulPartitionedCallб-batch_normalization_8/StatefulPartitionedCallб-batch_normalization_9/StatefulPartitionedCallб1spectral_normalization_16/StatefulPartitionedCallб1spectral_normalization_17/StatefulPartitionedCallб1spectral_normalization_18/StatefulPartitionedCallб1spectral_normalization_19/StatefulPartitionedCallК
reshape_2/PartitionedCallPartitionedCallinputs*
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
GPU2*0J 8ѓ *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_2461111Ђ
up_sampling2d_6/PartitionedCallPartitionedCall"reshape_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_2460793Ш
1spectral_normalization_16/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_6/PartitionedCall:output:0!spectral_normalization_16_2461126!spectral_normalization_16_2461128*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *_
fZRX
V__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_2461125║
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_16/StatefulPartitionedCall:output:0batch_normalization_8_2461131batch_normalization_8_2461133batch_normalization_8_2461135batch_normalization_8_2461137*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2460818ћ
up_sampling2d_7/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_2460876Ш
1spectral_normalization_17/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_7/PartitionedCall:output:0!spectral_normalization_17_2461153!spectral_normalization_17_2461155*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *_
fZRX
V__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_2461152║
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_17/StatefulPartitionedCall:output:0batch_normalization_9_2461158batch_normalization_9_2461160batch_normalization_9_2461162batch_normalization_9_2461164*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2460901ћ
up_sampling2d_8/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_2460959Ш
1spectral_normalization_18/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_8/PartitionedCall:output:0!spectral_normalization_18_2461180!spectral_normalization_18_2461182*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *_
fZRX
V__inference_spectral_normalization_18_layer_call_and_return_conditional_losses_2461179└
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_18/StatefulPartitionedCall:output:0batch_normalization_10_2461185batch_normalization_10_2461187batch_normalization_10_2461189batch_normalization_10_2461191*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2460984ѕ
/tf.__operators__.getitem_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ъ
)tf.__operators__.getitem_13/strided_sliceStridedSlice7batch_normalization_10/StatefulPartitionedCall:output:08tf.__operators__.getitem_13/strided_slice/stack:output:0:tf.__operators__.getitem_13/strided_slice/stack_1:output:0:tf.__operators__.getitem_13/strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+                           *

begin_mask*
end_maskl
tf.reverse_2/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB"      ╔
tf.reverse_2/ReverseV2	ReverseV22tf.__operators__.getitem_13/strided_slice:output:0$tf.reverse_2/ReverseV2/axis:output:0*
T0*A
_output_shapes/
-:+                           ѕ
/tf.__operators__.getitem_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                і
1tf.__operators__.getitem_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ъ
)tf.__operators__.getitem_12/strided_sliceStridedSlice7batch_normalization_10/StatefulPartitionedCall:output:08tf.__operators__.getitem_12/strided_slice/stack:output:0:tf.__operators__.getitem_12/strided_slice/stack_1:output:0:tf.__operators__.getitem_12/strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+                           *

begin_mask*
end_maskѕ
/tf.__operators__.getitem_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"        *       і
1tf.__operators__.getitem_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ъ
)tf.__operators__.getitem_14/strided_sliceStridedSlice7batch_normalization_10/StatefulPartitionedCall:output:08tf.__operators__.getitem_14/strided_slice/stack:output:0:tf.__operators__.getitem_14/strided_slice/stack_1:output:0:tf.__operators__.getitem_14/strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+                           *

begin_mask*
end_maskѕ
/tf.__operators__.getitem_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                і
1tf.__operators__.getitem_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Є
)tf.__operators__.getitem_15/strided_sliceStridedSlicetf.reverse_2/ReverseV2:output:08tf.__operators__.getitem_15/strided_slice/stack:output:0:tf.__operators__.getitem_15/strided_slice/stack_1:output:0:tf.__operators__.getitem_15/strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+                           *

begin_mask*
end_maskѕ
/tf.__operators__.getitem_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"        0       і
1tf.__operators__.getitem_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"        3       і
1tf.__operators__.getitem_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ъ
)tf.__operators__.getitem_16/strided_sliceStridedSlice7batch_normalization_10/StatefulPartitionedCall:output:08tf.__operators__.getitem_16/strided_slice/stack:output:0:tf.__operators__.getitem_16/strided_slice/stack_1:output:0:tf.__operators__.getitem_16/strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+                           *

begin_mask*
end_maskѕ
/tf.__operators__.getitem_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"        	       і
1tf.__operators__.getitem_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                і
1tf.__operators__.getitem_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Є
)tf.__operators__.getitem_17/strided_sliceStridedSlicetf.reverse_2/ReverseV2:output:08tf.__operators__.getitem_17/strided_slice/stack:output:0:tf.__operators__.getitem_17/strided_slice/stack_1:output:0:tf.__operators__.getitem_17/strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+                           *

begin_mask*
end_maskY
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :═
tf.concat_2/concatConcatV22tf.__operators__.getitem_12/strided_slice:output:02tf.__operators__.getitem_13/strided_slice:output:02tf.__operators__.getitem_14/strided_slice:output:02tf.__operators__.getitem_15/strided_slice:output:02tf.__operators__.getitem_16/strided_slice:output:02tf.__operators__.getitem_17/strided_slice:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+                           ж
1spectral_normalization_19/StatefulPartitionedCallStatefulPartitionedCalltf.concat_2/concat:output:0!spectral_normalization_19_2461234!spectral_normalization_19_2461236*
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
GPU2*0J 8ѓ *_
fZRX
V__inference_spectral_normalization_19_layer_call_and_return_conditional_losses_2461233└
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_19/StatefulPartitionedCall:output:0batch_normalization_11_2461239batch_normalization_11_2461241batch_normalization_11_2461243batch_normalization_11_2461245*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2461048а
IdentityIdentity7batch_normalization_11/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           п
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall2^spectral_normalization_16/StatefulPartitionedCall2^spectral_normalization_17/StatefulPartitionedCall2^spectral_normalization_18/StatefulPartitionedCall2^spectral_normalization_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:         ђ: : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2f
1spectral_normalization_16/StatefulPartitionedCall1spectral_normalization_16/StatefulPartitionedCall2f
1spectral_normalization_17/StatefulPartitionedCall1spectral_normalization_17/StatefulPartitionedCall2f
1spectral_normalization_18/StatefulPartitionedCall1spectral_normalization_18/StatefulPartitionedCall2f
1spectral_normalization_19/StatefulPartitionedCall1spectral_normalization_19/StatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Р
и
V__inference_spectral_normalization_18_layer_call_and_return_conditional_losses_2462945

inputsB
(conv2d_16_conv2d_readvariableop_resource: 7
)conv2d_16_biasadd_readvariableop_resource:
identityѕб conv2d_16/BiasAdd/ReadVariableOpбconv2d_16/Conv2D/ReadVariableOpљ
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0┐
conv2d_16/Conv2DConv2Dinputs'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
є
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Г
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           ~
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*A
_output_shapes/
-:+                           Ё
IdentityIdentityconv2d_16/Relu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           І
NoOpNoOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                            : : 2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
ў	
М
8__inference_batch_normalization_10_layer_call_fn_2462998

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2460984Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ћ
h
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_2460876

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:х
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(ў
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
м
b
F__inference_reshape_2_layer_call_and_return_conditional_losses_2461111

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
Њ8
Е
V__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_2462835

inputs9
reshape_readvariableop_resource:@ C
1spectral_normalize_matmul_readvariableop_resource: 7
)conv2d_15_biasadd_readvariableop_resource: 
identityѕбReshape/ReadVariableOpб conv2d_15/BiasAdd/ReadVariableOpбconv2d_15/Conv2D/ReadVariableOpб#spectral_normalize/AssignVariableOpб%spectral_normalize/AssignVariableOp_1б(spectral_normalize/MatMul/ReadVariableOpб!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:@ *
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"        t
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	ђ џ
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

: *
dtype0ц
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes
:	ђ*
transpose_b(
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	ђv
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
:	ђЁ
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes

: ѓ
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

: x
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

: 
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

: ђ
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	ђї
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes

: ▓
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
:@ *
dtype0е
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:@ y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @       А
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:@ ш
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(»
conv2d_15/Conv2D/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
:@ *
dtype0┐
conv2d_15/Conv2DConv2Dinputs'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
є
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Г
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            ~
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*A
_output_shapes/
-:+                            Ё
IdentityIdentityconv2d_15/Relu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                            ┴
NoOpNoOp^Reshape/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                           @: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ўu
»
#__inference__traced_restore_2463381
file_prefixL
1assignvariableop_spectral_normalization_16_kernel:	ђ@C
1assignvariableop_1_spectral_normalization_16_sn_u:@<
.assignvariableop_2_batch_normalization_8_gamma:@;
-assignvariableop_3_batch_normalization_8_beta:@B
4assignvariableop_4_batch_normalization_8_moving_mean:@F
8assignvariableop_5_batch_normalization_8_moving_variance:@M
3assignvariableop_6_spectral_normalization_17_kernel:@ C
1assignvariableop_7_spectral_normalization_17_sn_u: <
.assignvariableop_8_batch_normalization_9_gamma: ;
-assignvariableop_9_batch_normalization_9_beta: C
5assignvariableop_10_batch_normalization_9_moving_mean: G
9assignvariableop_11_batch_normalization_9_moving_variance: N
4assignvariableop_12_spectral_normalization_18_kernel: D
2assignvariableop_13_spectral_normalization_18_sn_u:>
0assignvariableop_14_batch_normalization_10_gamma:=
/assignvariableop_15_batch_normalization_10_beta:D
6assignvariableop_16_batch_normalization_10_moving_mean:H
:assignvariableop_17_batch_normalization_10_moving_variance:N
4assignvariableop_18_spectral_normalization_19_kernel:D
2assignvariableop_19_spectral_normalization_19_sn_u:>
0assignvariableop_20_batch_normalization_11_gamma:=
/assignvariableop_21_batch_normalization_11_beta:D
6assignvariableop_22_batch_normalization_11_moving_mean:H
:assignvariableop_23_batch_normalization_11_moving_variance:@
2assignvariableop_24_spectral_normalization_16_bias:@@
2assignvariableop_25_spectral_normalization_17_bias: @
2assignvariableop_26_spectral_normalization_18_bias:@
2assignvariableop_27_spectral_normalization_19_bias:
identity_29ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9 
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ц
valueЏBўB1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/sn_u/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-2/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/sn_u/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-4/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/sn_u/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-6/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/sn_u/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHф
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B ░
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ѕ
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOpAssignVariableOp1assignvariableop_spectral_normalization_16_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_1AssignVariableOp1assignvariableop_1_spectral_normalization_16_sn_uIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_8_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_8_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_8_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_8_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_6AssignVariableOp3assignvariableop_6_spectral_normalization_17_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_7AssignVariableOp1assignvariableop_7_spectral_normalization_17_sn_uIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_9_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_9_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_9_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_9_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_12AssignVariableOp4assignvariableop_12_spectral_normalization_18_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_13AssignVariableOp2assignvariableop_13_spectral_normalization_18_sn_uIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:А
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_10_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_10_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_10_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_10_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_18AssignVariableOp4assignvariableop_18_spectral_normalization_19_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_19AssignVariableOp2assignvariableop_19_spectral_normalization_19_sn_uIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:А
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_11_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_11_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_11_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_11_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_24AssignVariableOp2assignvariableop_24_spectral_normalization_16_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_25AssignVariableOp2assignvariableop_25_spectral_normalization_17_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_26AssignVariableOp2assignvariableop_26_spectral_normalization_18_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_27AssignVariableOp2assignvariableop_27_spectral_normalization_19_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 и
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_29IdentityIdentity_28:output:0^NoOp_1*
T0*
_output_shapes
: ц
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_29Identity_29:output:0*M
_input_shapes<
:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_27AssignVariableOp_272(
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
нu
Є
D__inference_model_2_layer_call_and_return_conditional_losses_2461917
input_5<
!spectral_normalization_16_2461830:	ђ@/
!spectral_normalization_16_2461832:@+
batch_normalization_8_2461835:@+
batch_normalization_8_2461837:@+
batch_normalization_8_2461839:@+
batch_normalization_8_2461841:@;
!spectral_normalization_17_2461845:@ /
!spectral_normalization_17_2461847: +
batch_normalization_9_2461850: +
batch_normalization_9_2461852: +
batch_normalization_9_2461854: +
batch_normalization_9_2461856: ;
!spectral_normalization_18_2461860: /
!spectral_normalization_18_2461862:,
batch_normalization_10_2461865:,
batch_normalization_10_2461867:,
batch_normalization_10_2461869:,
batch_normalization_10_2461871:;
!spectral_normalization_19_2461902:/
!spectral_normalization_19_2461904:,
batch_normalization_11_2461907:,
batch_normalization_11_2461909:,
batch_normalization_11_2461911:,
batch_normalization_11_2461913:
identityѕб.batch_normalization_10/StatefulPartitionedCallб.batch_normalization_11/StatefulPartitionedCallб-batch_normalization_8/StatefulPartitionedCallб-batch_normalization_9/StatefulPartitionedCallб1spectral_normalization_16/StatefulPartitionedCallб1spectral_normalization_17/StatefulPartitionedCallб1spectral_normalization_18/StatefulPartitionedCallб1spectral_normalization_19/StatefulPartitionedCall╚
reshape_2/PartitionedCallPartitionedCallinput_5*
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
GPU2*0J 8ѓ *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_2461111Ђ
up_sampling2d_6/PartitionedCallPartitionedCall"reshape_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_2460793Ш
1spectral_normalization_16/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_6/PartitionedCall:output:0!spectral_normalization_16_2461830!spectral_normalization_16_2461832*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *_
fZRX
V__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_2461125║
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_16/StatefulPartitionedCall:output:0batch_normalization_8_2461835batch_normalization_8_2461837batch_normalization_8_2461839batch_normalization_8_2461841*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2460818ћ
up_sampling2d_7/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_2460876Ш
1spectral_normalization_17/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_7/PartitionedCall:output:0!spectral_normalization_17_2461845!spectral_normalization_17_2461847*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *_
fZRX
V__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_2461152║
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_17/StatefulPartitionedCall:output:0batch_normalization_9_2461850batch_normalization_9_2461852batch_normalization_9_2461854batch_normalization_9_2461856*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2460901ћ
up_sampling2d_8/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_2460959Ш
1spectral_normalization_18/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_8/PartitionedCall:output:0!spectral_normalization_18_2461860!spectral_normalization_18_2461862*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *_
fZRX
V__inference_spectral_normalization_18_layer_call_and_return_conditional_losses_2461179└
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_18/StatefulPartitionedCall:output:0batch_normalization_10_2461865batch_normalization_10_2461867batch_normalization_10_2461869batch_normalization_10_2461871*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2460984ѕ
/tf.__operators__.getitem_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ъ
)tf.__operators__.getitem_13/strided_sliceStridedSlice7batch_normalization_10/StatefulPartitionedCall:output:08tf.__operators__.getitem_13/strided_slice/stack:output:0:tf.__operators__.getitem_13/strided_slice/stack_1:output:0:tf.__operators__.getitem_13/strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+                           *

begin_mask*
end_maskl
tf.reverse_2/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB"      ╔
tf.reverse_2/ReverseV2	ReverseV22tf.__operators__.getitem_13/strided_slice:output:0$tf.reverse_2/ReverseV2/axis:output:0*
T0*A
_output_shapes/
-:+                           ѕ
/tf.__operators__.getitem_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                і
1tf.__operators__.getitem_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ъ
)tf.__operators__.getitem_12/strided_sliceStridedSlice7batch_normalization_10/StatefulPartitionedCall:output:08tf.__operators__.getitem_12/strided_slice/stack:output:0:tf.__operators__.getitem_12/strided_slice/stack_1:output:0:tf.__operators__.getitem_12/strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+                           *

begin_mask*
end_maskѕ
/tf.__operators__.getitem_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"        *       і
1tf.__operators__.getitem_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ъ
)tf.__operators__.getitem_14/strided_sliceStridedSlice7batch_normalization_10/StatefulPartitionedCall:output:08tf.__operators__.getitem_14/strided_slice/stack:output:0:tf.__operators__.getitem_14/strided_slice/stack_1:output:0:tf.__operators__.getitem_14/strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+                           *

begin_mask*
end_maskѕ
/tf.__operators__.getitem_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                і
1tf.__operators__.getitem_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Є
)tf.__operators__.getitem_15/strided_sliceStridedSlicetf.reverse_2/ReverseV2:output:08tf.__operators__.getitem_15/strided_slice/stack:output:0:tf.__operators__.getitem_15/strided_slice/stack_1:output:0:tf.__operators__.getitem_15/strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+                           *

begin_mask*
end_maskѕ
/tf.__operators__.getitem_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"        0       і
1tf.__operators__.getitem_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"        3       і
1tf.__operators__.getitem_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ъ
)tf.__operators__.getitem_16/strided_sliceStridedSlice7batch_normalization_10/StatefulPartitionedCall:output:08tf.__operators__.getitem_16/strided_slice/stack:output:0:tf.__operators__.getitem_16/strided_slice/stack_1:output:0:tf.__operators__.getitem_16/strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+                           *

begin_mask*
end_maskѕ
/tf.__operators__.getitem_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"        	       і
1tf.__operators__.getitem_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                і
1tf.__operators__.getitem_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Є
)tf.__operators__.getitem_17/strided_sliceStridedSlicetf.reverse_2/ReverseV2:output:08tf.__operators__.getitem_17/strided_slice/stack:output:0:tf.__operators__.getitem_17/strided_slice/stack_1:output:0:tf.__operators__.getitem_17/strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+                           *

begin_mask*
end_maskY
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :═
tf.concat_2/concatConcatV22tf.__operators__.getitem_12/strided_slice:output:02tf.__operators__.getitem_13/strided_slice:output:02tf.__operators__.getitem_14/strided_slice:output:02tf.__operators__.getitem_15/strided_slice:output:02tf.__operators__.getitem_16/strided_slice:output:02tf.__operators__.getitem_17/strided_slice:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+                           ж
1spectral_normalization_19/StatefulPartitionedCallStatefulPartitionedCalltf.concat_2/concat:output:0!spectral_normalization_19_2461902!spectral_normalization_19_2461904*
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
GPU2*0J 8ѓ *_
fZRX
V__inference_spectral_normalization_19_layer_call_and_return_conditional_losses_2461233└
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_19/StatefulPartitionedCall:output:0batch_normalization_11_2461907batch_normalization_11_2461909batch_normalization_11_2461911batch_normalization_11_2461913*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2461048а
IdentityIdentity7batch_normalization_11/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           п
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall2^spectral_normalization_16/StatefulPartitionedCall2^spectral_normalization_17/StatefulPartitionedCall2^spectral_normalization_18/StatefulPartitionedCall2^spectral_normalization_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:         ђ: : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2f
1spectral_normalization_16/StatefulPartitionedCall1spectral_normalization_16/StatefulPartitionedCall2f
1spectral_normalization_17/StatefulPartitionedCall1spectral_normalization_17/StatefulPartitionedCall2f
1spectral_normalization_18/StatefulPartitionedCall1spectral_normalization_18/StatefulPartitionedCall2f
1spectral_normalization_19/StatefulPartitionedCall1spectral_normalization_19/StatefulPartitionedCall:Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_5
Т
И
V__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_2461125

inputsC
(conv2d_14_conv2d_readvariableop_resource:	ђ@7
)conv2d_14_biasadd_readvariableop_resource:@
identityѕб conv2d_14/BiasAdd/ReadVariableOpбconv2d_14/Conv2D/ReadVariableOpЉ
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*'
_output_shapes
:	ђ@*
dtype0┐
conv2d_14/Conv2DConv2Dinputs'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
є
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Г
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @~
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @Ё
IdentityIdentityconv2d_14/Relu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           @І
NoOpNoOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,                           ђ: : 2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Р
и
V__inference_spectral_normalization_18_layer_call_and_return_conditional_losses_2461179

inputsB
(conv2d_16_conv2d_readvariableop_resource: 7
)conv2d_16_biasadd_readvariableop_resource:
identityѕб conv2d_16/BiasAdd/ReadVariableOpбconv2d_16/Conv2D/ReadVariableOpљ
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0┐
conv2d_16/Conv2DConv2Dinputs'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
є
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Г
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           ~
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*A
_output_shapes/
-:+                           Ё
IdentityIdentityconv2d_16/Relu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           І
NoOpNoOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                            : : 2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Є
┴
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2460932

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
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
-:+                            н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
ѕ
┬
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2463180

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
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
-:+                           н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
═
Ю
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2460818

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
═
Ю
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2462729

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Њ8
Е
V__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_2461474

inputs9
reshape_readvariableop_resource:@ C
1spectral_normalize_matmul_readvariableop_resource: 7
)conv2d_15_biasadd_readvariableop_resource: 
identityѕбReshape/ReadVariableOpб conv2d_15/BiasAdd/ReadVariableOpбconv2d_15/Conv2D/ReadVariableOpб#spectral_normalize/AssignVariableOpб%spectral_normalize/AssignVariableOp_1б(spectral_normalize/MatMul/ReadVariableOpб!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:@ *
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"        t
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	ђ џ
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

: *
dtype0ц
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes
:	ђ*
transpose_b(
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	ђv
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
:	ђЁ
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes

: ѓ
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

: x
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

: 
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

: ђ
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	ђї
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes

: ▓
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
:@ *
dtype0е
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:@ y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @       А
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:@ ш
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(»
conv2d_15/Conv2D/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
:@ *
dtype0┐
conv2d_15/Conv2DConv2Dinputs'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
є
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Г
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            ~
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*A
_output_shapes/
-:+                            Ё
IdentityIdentityconv2d_15/Relu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                            ┴
NoOpNoOp^Reshape/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                           @: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ћ	
м
7__inference_batch_normalization_8_layer_call_fn_2462711

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2460849Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ћ
h
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_2460793

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:х
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(ў
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
И
G
+__inference_reshape_2_layer_call_fn_2462583

inputs
identityй
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
GPU2*0J 8ѓ *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_2461111i
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
ё	
═
;__inference_spectral_normalization_17_layer_call_fn_2462784

inputs!
unknown:@ 
	unknown_0: 
	unknown_1: 
identityѕбStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *_
fZRX
V__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_2461474Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                           @: : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ѓ8
Е
V__inference_spectral_normalization_19_layer_call_and_return_conditional_losses_2463118

inputs9
reshape_readvariableop_resource:C
1spectral_normalize_matmul_readvariableop_resource:7
)conv2d_17_biasadd_readvariableop_resource:
identityѕбReshape/ReadVariableOpб conv2d_17/BiasAdd/ReadVariableOpбconv2d_17/Conv2D/ReadVariableOpб#spectral_normalize/AssignVariableOpб%spectral_normalize/AssignVariableOp_1б(spectral_normalize/MatMul/ReadVariableOpб!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:*
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       s
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes

:џ
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Б
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes

:*
transpose_b(~
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes

:v
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

:Ё
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes

:ѓ
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:x
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

:
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes

:ї
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes

:▓
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
:*
dtype0е
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            А
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:ш
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(»
conv2d_17/Conv2D/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
:*
dtype0┐
conv2d_17/Conv2DConv2Dinputs'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
є
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Г
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           ~
conv2d_17/TanhTanhconv2d_17/BiasAdd:output:0*
T0*A
_output_shapes/
-:+                           {
IdentityIdentityconv2d_17/Tanh:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ┴
NoOpNoOp^Reshape/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                           : : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ї8
Е
V__inference_spectral_normalization_18_layer_call_and_return_conditional_losses_2462985

inputs9
reshape_readvariableop_resource: C
1spectral_normalize_matmul_readvariableop_resource:7
)conv2d_16_biasadd_readvariableop_resource:
identityѕбReshape/ReadVariableOpб conv2d_16/BiasAdd/ReadVariableOpбconv2d_16/Conv2D/ReadVariableOpб#spectral_normalize/AssignVariableOpб%spectral_normalize/AssignVariableOp_1б(spectral_normalize/MatMul/ReadVariableOpб!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
: *
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       s
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes

:`џ
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Б
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

:`Ё
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes

:ѓ
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:x
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

:
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes

:`ї
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes

:▓
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
: *
dtype0е
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
: y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             А
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
: ш
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(»
conv2d_16/Conv2D/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
: *
dtype0┐
conv2d_16/Conv2DConv2Dinputs'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
є
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Г
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           ~
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*A
_output_shapes/
-:+                           Ё
IdentityIdentityconv2d_16/Relu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           ┴
NoOpNoOp^Reshape/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                            : : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
й
M
1__inference_up_sampling2d_8_layer_call_fn_2462902

inputs
identityП
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_2460959Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ћ
h
L__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_2460959

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:х
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(ў
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Џ8
ф
V__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_2462685

inputs:
reshape_readvariableop_resource:	ђ@C
1spectral_normalize_matmul_readvariableop_resource:@7
)conv2d_14_biasadd_readvariableop_resource:@
identityѕбReshape/ReadVariableOpб conv2d_14/BiasAdd/ReadVariableOpбconv2d_14/Conv2D/ReadVariableOpб#spectral_normalize/AssignVariableOpб%spectral_normalize/AssignVariableOp_1б(spectral_normalize/MatMul/ReadVariableOpб!spectral_normalize/ReadVariableOp
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*'
_output_shapes
:	ђ@*
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   t
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	ђ$@џ
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0ц
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes
:	ђ$*
transpose_b(
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	ђ$v
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
:	ђ$Ё
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes

:@ѓ
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:@x
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

:@
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:@ђ
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	ђ$ї
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes

:@▓
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
:	ђ@*
dtype0Е
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*'
_output_shapes
:	ђ@y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   	   ђ   @   б
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*'
_output_shapes
:	ђ@ш
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(░
conv2d_14/Conv2D/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*'
_output_shapes
:	ђ@*
dtype0┐
conv2d_14/Conv2DConv2Dinputs'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
є
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Г
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @~
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @Ё
IdentityIdentityconv2d_14/Relu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           @┴
NoOpNoOp^Reshape/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,                           ђ: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
й
M
1__inference_up_sampling2d_6_layer_call_fn_2462602

inputs
identityП
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_2460793Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ћ
h
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_2462614

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:х
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(ў
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Є	
╬
;__inference_spectral_normalization_16_layer_call_fn_2462634

inputs"
unknown:	ђ@
	unknown_0:@
	unknown_1:@
identityѕбStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *_
fZRX
V__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_2461535Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,                           ђ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Є
┴
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2462747

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
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
-:+                           @н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Є
┴
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2460849

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
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
-:+                           @н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
О
и
V__inference_spectral_normalization_19_layer_call_and_return_conditional_losses_2463078

inputsB
(conv2d_17_conv2d_readvariableop_resource:7
)conv2d_17_biasadd_readvariableop_resource:
identityѕб conv2d_17/BiasAdd/ReadVariableOpбconv2d_17/Conv2D/ReadVariableOpљ
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0┐
conv2d_17/Conv2DConv2Dinputs'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
є
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Г
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           ~
conv2d_17/TanhTanhconv2d_17/BiasAdd:output:0*
T0*A
_output_shapes/
-:+                           {
IdentityIdentityconv2d_17/Tanh:y:0^NoOp*
T0*A
_output_shapes/
-:+                           І
NoOpNoOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
┌
░
;__inference_spectral_normalization_19_layer_call_fn_2463056

inputs!
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallѕ
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
GPU2*0J 8ѓ *_
fZRX
V__inference_spectral_normalization_19_layer_call_and_return_conditional_losses_2461233Ѕ
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
1:+                           : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
м
b
F__inference_reshape_2_layer_call_and_return_conditional_losses_2462597

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
й
M
1__inference_up_sampling2d_7_layer_call_fn_2462752

inputs
identityП
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_2460876Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ф┬
ї
D__inference_model_2_layer_call_and_return_conditional_losses_2462324

inputs]
Bspectral_normalization_16_conv2d_14_conv2d_readvariableop_resource:	ђ@Q
Cspectral_normalization_16_conv2d_14_biasadd_readvariableop_resource:@;
-batch_normalization_8_readvariableop_resource:@=
/batch_normalization_8_readvariableop_1_resource:@L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:@\
Bspectral_normalization_17_conv2d_15_conv2d_readvariableop_resource:@ Q
Cspectral_normalization_17_conv2d_15_biasadd_readvariableop_resource: ;
-batch_normalization_9_readvariableop_resource: =
/batch_normalization_9_readvariableop_1_resource: L
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource: \
Bspectral_normalization_18_conv2d_16_conv2d_readvariableop_resource: Q
Cspectral_normalization_18_conv2d_16_biasadd_readvariableop_resource:<
.batch_normalization_10_readvariableop_resource:>
0batch_normalization_10_readvariableop_1_resource:M
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:\
Bspectral_normalization_19_conv2d_17_conv2d_readvariableop_resource:Q
Cspectral_normalization_19_conv2d_17_biasadd_readvariableop_resource:<
.batch_normalization_11_readvariableop_resource:>
0batch_normalization_11_readvariableop_1_resource:M
?batch_normalization_11_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:
identityѕб6batch_normalization_10/FusedBatchNormV3/ReadVariableOpб8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_10/ReadVariableOpб'batch_normalization_10/ReadVariableOp_1б6batch_normalization_11/FusedBatchNormV3/ReadVariableOpб8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_11/ReadVariableOpб'batch_normalization_11/ReadVariableOp_1б5batch_normalization_8/FusedBatchNormV3/ReadVariableOpб7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_8/ReadVariableOpб&batch_normalization_8/ReadVariableOp_1б5batch_normalization_9/FusedBatchNormV3/ReadVariableOpб7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_9/ReadVariableOpб&batch_normalization_9/ReadVariableOp_1б:spectral_normalization_16/conv2d_14/BiasAdd/ReadVariableOpб9spectral_normalization_16/conv2d_14/Conv2D/ReadVariableOpб:spectral_normalization_17/conv2d_15/BiasAdd/ReadVariableOpб9spectral_normalization_17/conv2d_15/Conv2D/ReadVariableOpб:spectral_normalization_18/conv2d_16/BiasAdd/ReadVariableOpб9spectral_normalization_18/conv2d_16/Conv2D/ReadVariableOpб:spectral_normalization_19/conv2d_17/BiasAdd/ReadVariableOpб9spectral_normalization_19/conv2d_17/Conv2D/ReadVariableOpE
reshape_2/ShapeShapeinputs*
T0*
_output_shapes
:g
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :\
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :ђ█
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ђ
reshape_2/ReshapeReshapeinputs reshape_2/Reshape/shape:output:0*
T0*0
_output_shapes
:         ђf
up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Ђ
up_sampling2d_6/mulMulup_sampling2d_6/Const:output:0 up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:¤
,up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighborreshape_2/Reshape:output:0up_sampling2d_6/mul:z:0*
T0*0
_output_shapes
:         ђ*
half_pixel_centers(┼
9spectral_normalization_16/conv2d_14/Conv2D/ReadVariableOpReadVariableOpBspectral_normalization_16_conv2d_14_conv2d_readvariableop_resource*'
_output_shapes
:	ђ@*
dtype0ў
*spectral_normalization_16/conv2d_14/Conv2DConv2D=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0Aspectral_normalization_16/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
║
:spectral_normalization_16/conv2d_14/BiasAdd/ReadVariableOpReadVariableOpCspectral_normalization_16_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ж
+spectral_normalization_16/conv2d_14/BiasAddBiasAdd3spectral_normalization_16/conv2d_14/Conv2D:output:0Bspectral_normalization_16/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @а
(spectral_normalization_16/conv2d_14/ReluRelu4spectral_normalization_16/conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:         @ј
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:@*
dtype0њ
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:@*
dtype0░
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0┤
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0н
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV36spectral_normalization_16/conv2d_14/Relu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( f
up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Ђ
up_sampling2d_7/mulMulup_sampling2d_7/Const:output:0 up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:я
,up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor*batch_normalization_8/FusedBatchNormV3:y:0up_sampling2d_7/mul:z:0*
T0*/
_output_shapes
:         @*
half_pixel_centers(─
9spectral_normalization_17/conv2d_15/Conv2D/ReadVariableOpReadVariableOpBspectral_normalization_17_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0ў
*spectral_normalization_17/conv2d_15/Conv2DConv2D=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0Aspectral_normalization_17/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
║
:spectral_normalization_17/conv2d_15/BiasAdd/ReadVariableOpReadVariableOpCspectral_normalization_17_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ж
+spectral_normalization_17/conv2d_15/BiasAddBiasAdd3spectral_normalization_17/conv2d_15/Conv2D:output:0Bspectral_normalization_17/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          а
(spectral_normalization_17/conv2d_15/ReluRelu4spectral_normalization_17/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:          ј
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype0њ
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype0░
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0┤
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0н
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV36spectral_normalization_17/conv2d_15/Relu:activations:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( f
up_sampling2d_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Ђ
up_sampling2d_8/mulMulup_sampling2d_8/Const:output:0 up_sampling2d_8/Const_1:output:0*
T0*
_output_shapes
:я
,up_sampling2d_8/resize/ResizeNearestNeighborResizeNearestNeighbor*batch_normalization_9/FusedBatchNormV3:y:0up_sampling2d_8/mul:z:0*
T0*/
_output_shapes
:         < *
half_pixel_centers(─
9spectral_normalization_18/conv2d_16/Conv2D/ReadVariableOpReadVariableOpBspectral_normalization_18_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ў
*spectral_normalization_18/conv2d_16/Conv2DConv2D=up_sampling2d_8/resize/ResizeNearestNeighbor:resized_images:0Aspectral_normalization_18/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <*
paddingSAME*
strides
║
:spectral_normalization_18/conv2d_16/BiasAdd/ReadVariableOpReadVariableOpCspectral_normalization_18_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ж
+spectral_normalization_18/conv2d_16/BiasAddBiasAdd3spectral_normalization_18/conv2d_16/Conv2D:output:0Bspectral_normalization_18/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <а
(spectral_normalization_18/conv2d_16/ReluRelu4spectral_normalization_18/conv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:         <љ
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
:*
dtype0ћ
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
:*
dtype0▓
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Х
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0┘
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV36spectral_normalization_18/conv2d_16/Relu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         <:::::*
epsilon%oЃ:*
is_training( ѕ
/tf.__operators__.getitem_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ђ
)tf.__operators__.getitem_13/strided_sliceStridedSlice+batch_normalization_10/FusedBatchNormV3:y:08tf.__operators__.getitem_13/strided_slice/stack:output:0:tf.__operators__.getitem_13/strided_slice/stack_1:output:0:tf.__operators__.getitem_13/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_maskl
tf.reverse_2/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB"      и
tf.reverse_2/ReverseV2	ReverseV22tf.__operators__.getitem_13/strided_slice:output:0$tf.reverse_2/ReverseV2/axis:output:0*
T0*/
_output_shapes
:         ѕ
/tf.__operators__.getitem_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                і
1tf.__operators__.getitem_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ђ
)tf.__operators__.getitem_12/strided_sliceStridedSlice+batch_normalization_10/FusedBatchNormV3:y:08tf.__operators__.getitem_12/strided_slice/stack:output:0:tf.__operators__.getitem_12/strided_slice/stack_1:output:0:tf.__operators__.getitem_12/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_maskѕ
/tf.__operators__.getitem_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"        *       і
1tf.__operators__.getitem_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ђ
)tf.__operators__.getitem_14/strided_sliceStridedSlice+batch_normalization_10/FusedBatchNormV3:y:08tf.__operators__.getitem_14/strided_slice/stack:output:0:tf.__operators__.getitem_14/strided_slice/stack_1:output:0:tf.__operators__.getitem_14/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_maskѕ
/tf.__operators__.getitem_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                і
1tf.__operators__.getitem_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ш
)tf.__operators__.getitem_15/strided_sliceStridedSlicetf.reverse_2/ReverseV2:output:08tf.__operators__.getitem_15/strided_slice/stack:output:0:tf.__operators__.getitem_15/strided_slice/stack_1:output:0:tf.__operators__.getitem_15/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_maskѕ
/tf.__operators__.getitem_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"        0       і
1tf.__operators__.getitem_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"        3       і
1tf.__operators__.getitem_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ђ
)tf.__operators__.getitem_16/strided_sliceStridedSlice+batch_normalization_10/FusedBatchNormV3:y:08tf.__operators__.getitem_16/strided_slice/stack:output:0:tf.__operators__.getitem_16/strided_slice/stack_1:output:0:tf.__operators__.getitem_16/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_maskѕ
/tf.__operators__.getitem_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"        	       і
1tf.__operators__.getitem_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                і
1tf.__operators__.getitem_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ш
)tf.__operators__.getitem_17/strided_sliceStridedSlicetf.reverse_2/ReverseV2:output:08tf.__operators__.getitem_17/strided_slice/stack:output:0:tf.__operators__.getitem_17/strided_slice/stack_1:output:0:tf.__operators__.getitem_17/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         	*

begin_mask*
end_maskY
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :╗
tf.concat_2/concatConcatV22tf.__operators__.getitem_12/strided_slice:output:02tf.__operators__.getitem_13/strided_slice:output:02tf.__operators__.getitem_14/strided_slice:output:02tf.__operators__.getitem_15/strided_slice:output:02tf.__operators__.getitem_16/strided_slice:output:02tf.__operators__.getitem_17/strided_slice:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*/
_output_shapes
:         <─
9spectral_normalization_19/conv2d_17/Conv2D/ReadVariableOpReadVariableOpBspectral_normalization_19_conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ш
*spectral_normalization_19/conv2d_17/Conv2DConv2Dtf.concat_2/concat:output:0Aspectral_normalization_19/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <*
paddingSAME*
strides
║
:spectral_normalization_19/conv2d_17/BiasAdd/ReadVariableOpReadVariableOpCspectral_normalization_19_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ж
+spectral_normalization_19/conv2d_17/BiasAddBiasAdd3spectral_normalization_19/conv2d_17/Conv2D:output:0Bspectral_normalization_19/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <а
(spectral_normalization_19/conv2d_17/TanhTanh4spectral_normalization_19/conv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:         <љ
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes
:*
dtype0ћ
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes
:*
dtype0▓
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Х
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0¤
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3,spectral_normalization_19/conv2d_17/Tanh:y:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         <:::::*
epsilon%oЃ:*
is_training( ѓ
IdentityIdentity+batch_normalization_11/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:         <║

NoOpNoOp7^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_17^batch_normalization_11/FusedBatchNormV3/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_11/ReadVariableOp(^batch_normalization_11/ReadVariableOp_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1;^spectral_normalization_16/conv2d_14/BiasAdd/ReadVariableOp:^spectral_normalization_16/conv2d_14/Conv2D/ReadVariableOp;^spectral_normalization_17/conv2d_15/BiasAdd/ReadVariableOp:^spectral_normalization_17/conv2d_15/Conv2D/ReadVariableOp;^spectral_normalization_18/conv2d_16/BiasAdd/ReadVariableOp:^spectral_normalization_18/conv2d_16/Conv2D/ReadVariableOp;^spectral_normalization_19/conv2d_17/BiasAdd/ReadVariableOp:^spectral_normalization_19/conv2d_17/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:         ђ: : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12p
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp6batch_normalization_11/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_18batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_11/ReadVariableOp%batch_normalization_11/ReadVariableOp2R
'batch_normalization_11/ReadVariableOp_1'batch_normalization_11/ReadVariableOp_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12x
:spectral_normalization_16/conv2d_14/BiasAdd/ReadVariableOp:spectral_normalization_16/conv2d_14/BiasAdd/ReadVariableOp2v
9spectral_normalization_16/conv2d_14/Conv2D/ReadVariableOp9spectral_normalization_16/conv2d_14/Conv2D/ReadVariableOp2x
:spectral_normalization_17/conv2d_15/BiasAdd/ReadVariableOp:spectral_normalization_17/conv2d_15/BiasAdd/ReadVariableOp2v
9spectral_normalization_17/conv2d_15/Conv2D/ReadVariableOp9spectral_normalization_17/conv2d_15/Conv2D/ReadVariableOp2x
:spectral_normalization_18/conv2d_16/BiasAdd/ReadVariableOp:spectral_normalization_18/conv2d_16/BiasAdd/ReadVariableOp2v
9spectral_normalization_18/conv2d_16/Conv2D/ReadVariableOp9spectral_normalization_18/conv2d_16/Conv2D/ReadVariableOp2x
:spectral_normalization_19/conv2d_17/BiasAdd/ReadVariableOp:spectral_normalization_19/conv2d_17/BiasAdd/ReadVariableOp2v
9spectral_normalization_19/conv2d_17/Conv2D/ReadVariableOp9spectral_normalization_19/conv2d_17/Conv2D/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ё	
═
;__inference_spectral_normalization_19_layer_call_fn_2463067

inputs!
unknown:
	unknown_0:
	unknown_1:
identityѕбStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *_
fZRX
V__inference_spectral_normalization_19_layer_call_and_return_conditional_losses_2461352Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                           : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ќ	
М
8__inference_batch_normalization_10_layer_call_fn_2463011

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2461015Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
│
Ш
)__inference_model_2_layer_call_fn_2462125

inputs"
unknown:	ђ@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identityѕбStatefulPartitionedCallА
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
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_2461249Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:         ђ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
со
в
"__inference__wrapped_model_2460777
input_5e
Jmodel_2_spectral_normalization_16_conv2d_14_conv2d_readvariableop_resource:	ђ@Y
Kmodel_2_spectral_normalization_16_conv2d_14_biasadd_readvariableop_resource:@C
5model_2_batch_normalization_8_readvariableop_resource:@E
7model_2_batch_normalization_8_readvariableop_1_resource:@T
Fmodel_2_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:@V
Hmodel_2_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:@d
Jmodel_2_spectral_normalization_17_conv2d_15_conv2d_readvariableop_resource:@ Y
Kmodel_2_spectral_normalization_17_conv2d_15_biasadd_readvariableop_resource: C
5model_2_batch_normalization_9_readvariableop_resource: E
7model_2_batch_normalization_9_readvariableop_1_resource: T
Fmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource: V
Hmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource: d
Jmodel_2_spectral_normalization_18_conv2d_16_conv2d_readvariableop_resource: Y
Kmodel_2_spectral_normalization_18_conv2d_16_biasadd_readvariableop_resource:D
6model_2_batch_normalization_10_readvariableop_resource:F
8model_2_batch_normalization_10_readvariableop_1_resource:U
Gmodel_2_batch_normalization_10_fusedbatchnormv3_readvariableop_resource:W
Imodel_2_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:d
Jmodel_2_spectral_normalization_19_conv2d_17_conv2d_readvariableop_resource:Y
Kmodel_2_spectral_normalization_19_conv2d_17_biasadd_readvariableop_resource:D
6model_2_batch_normalization_11_readvariableop_resource:F
8model_2_batch_normalization_11_readvariableop_1_resource:U
Gmodel_2_batch_normalization_11_fusedbatchnormv3_readvariableop_resource:W
Imodel_2_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:
identityѕб>model_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOpб@model_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1б-model_2/batch_normalization_10/ReadVariableOpб/model_2/batch_normalization_10/ReadVariableOp_1б>model_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOpб@model_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1б-model_2/batch_normalization_11/ReadVariableOpб/model_2/batch_normalization_11/ReadVariableOp_1б=model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOpб?model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1б,model_2/batch_normalization_8/ReadVariableOpб.model_2/batch_normalization_8/ReadVariableOp_1б=model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOpб?model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1б,model_2/batch_normalization_9/ReadVariableOpб.model_2/batch_normalization_9/ReadVariableOp_1бBmodel_2/spectral_normalization_16/conv2d_14/BiasAdd/ReadVariableOpбAmodel_2/spectral_normalization_16/conv2d_14/Conv2D/ReadVariableOpбBmodel_2/spectral_normalization_17/conv2d_15/BiasAdd/ReadVariableOpбAmodel_2/spectral_normalization_17/conv2d_15/Conv2D/ReadVariableOpбBmodel_2/spectral_normalization_18/conv2d_16/BiasAdd/ReadVariableOpбAmodel_2/spectral_normalization_18/conv2d_16/Conv2D/ReadVariableOpбBmodel_2/spectral_normalization_19/conv2d_17/BiasAdd/ReadVariableOpбAmodel_2/spectral_normalization_19/conv2d_17/Conv2D/ReadVariableOpN
model_2/reshape_2/ShapeShapeinput_5*
T0*
_output_shapes
:o
%model_2/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'model_2/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'model_2/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ф
model_2/reshape_2/strided_sliceStridedSlice model_2/reshape_2/Shape:output:0.model_2/reshape_2/strided_slice/stack:output:00model_2/reshape_2/strided_slice/stack_1:output:00model_2/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model_2/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :c
!model_2/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d
!model_2/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :ђЃ
model_2/reshape_2/Reshape/shapePack(model_2/reshape_2/strided_slice:output:0*model_2/reshape_2/Reshape/shape/1:output:0*model_2/reshape_2/Reshape/shape/2:output:0*model_2/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:њ
model_2/reshape_2/ReshapeReshapeinput_5(model_2/reshape_2/Reshape/shape:output:0*
T0*0
_output_shapes
:         ђn
model_2/up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      p
model_2/up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Ў
model_2/up_sampling2d_6/mulMul&model_2/up_sampling2d_6/Const:output:0(model_2/up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:у
4model_2/up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighbor"model_2/reshape_2/Reshape:output:0model_2/up_sampling2d_6/mul:z:0*
T0*0
_output_shapes
:         ђ*
half_pixel_centers(Н
Amodel_2/spectral_normalization_16/conv2d_14/Conv2D/ReadVariableOpReadVariableOpJmodel_2_spectral_normalization_16_conv2d_14_conv2d_readvariableop_resource*'
_output_shapes
:	ђ@*
dtype0░
2model_2/spectral_normalization_16/conv2d_14/Conv2DConv2DEmodel_2/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0Imodel_2/spectral_normalization_16/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
╩
Bmodel_2/spectral_normalization_16/conv2d_14/BiasAdd/ReadVariableOpReadVariableOpKmodel_2_spectral_normalization_16_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ђ
3model_2/spectral_normalization_16/conv2d_14/BiasAddBiasAdd;model_2/spectral_normalization_16/conv2d_14/Conv2D:output:0Jmodel_2/spectral_normalization_16/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @░
0model_2/spectral_normalization_16/conv2d_14/ReluRelu<model_2/spectral_normalization_16/conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:         @ъ
,model_2/batch_normalization_8/ReadVariableOpReadVariableOp5model_2_batch_normalization_8_readvariableop_resource*
_output_shapes
:@*
dtype0б
.model_2/batch_normalization_8/ReadVariableOp_1ReadVariableOp7model_2_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:@*
dtype0└
=model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0─
?model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ё
.model_2/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3>model_2/spectral_normalization_16/conv2d_14/Relu:activations:04model_2/batch_normalization_8/ReadVariableOp:value:06model_2/batch_normalization_8/ReadVariableOp_1:value:0Emodel_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( n
model_2/up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      p
model_2/up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Ў
model_2/up_sampling2d_7/mulMul&model_2/up_sampling2d_7/Const:output:0(model_2/up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:Ш
4model_2/up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor2model_2/batch_normalization_8/FusedBatchNormV3:y:0model_2/up_sampling2d_7/mul:z:0*
T0*/
_output_shapes
:         @*
half_pixel_centers(н
Amodel_2/spectral_normalization_17/conv2d_15/Conv2D/ReadVariableOpReadVariableOpJmodel_2_spectral_normalization_17_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0░
2model_2/spectral_normalization_17/conv2d_15/Conv2DConv2DEmodel_2/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0Imodel_2/spectral_normalization_17/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
╩
Bmodel_2/spectral_normalization_17/conv2d_15/BiasAdd/ReadVariableOpReadVariableOpKmodel_2_spectral_normalization_17_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ђ
3model_2/spectral_normalization_17/conv2d_15/BiasAddBiasAdd;model_2/spectral_normalization_17/conv2d_15/Conv2D:output:0Jmodel_2/spectral_normalization_17/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          ░
0model_2/spectral_normalization_17/conv2d_15/ReluRelu<model_2/spectral_normalization_17/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:          ъ
,model_2/batch_normalization_9/ReadVariableOpReadVariableOp5model_2_batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype0б
.model_2/batch_normalization_9/ReadVariableOp_1ReadVariableOp7model_2_batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype0└
=model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0─
?model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ё
.model_2/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3>model_2/spectral_normalization_17/conv2d_15/Relu:activations:04model_2/batch_normalization_9/ReadVariableOp:value:06model_2/batch_normalization_9/ReadVariableOp_1:value:0Emodel_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( n
model_2/up_sampling2d_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"      p
model_2/up_sampling2d_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Ў
model_2/up_sampling2d_8/mulMul&model_2/up_sampling2d_8/Const:output:0(model_2/up_sampling2d_8/Const_1:output:0*
T0*
_output_shapes
:Ш
4model_2/up_sampling2d_8/resize/ResizeNearestNeighborResizeNearestNeighbor2model_2/batch_normalization_9/FusedBatchNormV3:y:0model_2/up_sampling2d_8/mul:z:0*
T0*/
_output_shapes
:         < *
half_pixel_centers(н
Amodel_2/spectral_normalization_18/conv2d_16/Conv2D/ReadVariableOpReadVariableOpJmodel_2_spectral_normalization_18_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0░
2model_2/spectral_normalization_18/conv2d_16/Conv2DConv2DEmodel_2/up_sampling2d_8/resize/ResizeNearestNeighbor:resized_images:0Imodel_2/spectral_normalization_18/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <*
paddingSAME*
strides
╩
Bmodel_2/spectral_normalization_18/conv2d_16/BiasAdd/ReadVariableOpReadVariableOpKmodel_2_spectral_normalization_18_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ђ
3model_2/spectral_normalization_18/conv2d_16/BiasAddBiasAdd;model_2/spectral_normalization_18/conv2d_16/Conv2D:output:0Jmodel_2/spectral_normalization_18/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <░
0model_2/spectral_normalization_18/conv2d_16/ReluRelu<model_2/spectral_normalization_18/conv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:         <а
-model_2/batch_normalization_10/ReadVariableOpReadVariableOp6model_2_batch_normalization_10_readvariableop_resource*
_output_shapes
:*
dtype0ц
/model_2/batch_normalization_10/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_10_readvariableop_1_resource*
_output_shapes
:*
dtype0┬
>model_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0к
@model_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ѕ
/model_2/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3>model_2/spectral_normalization_18/conv2d_16/Relu:activations:05model_2/batch_normalization_10/ReadVariableOp:value:07model_2/batch_normalization_10/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         <:::::*
epsilon%oЃ:*
is_training( љ
7model_2/tf.__operators__.getitem_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               њ
9model_2/tf.__operators__.getitem_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               њ
9model_2/tf.__operators__.getitem_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Е
1model_2/tf.__operators__.getitem_13/strided_sliceStridedSlice3model_2/batch_normalization_10/FusedBatchNormV3:y:0@model_2/tf.__operators__.getitem_13/strided_slice/stack:output:0Bmodel_2/tf.__operators__.getitem_13/strided_slice/stack_1:output:0Bmodel_2/tf.__operators__.getitem_13/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_maskt
#model_2/tf.reverse_2/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB"      ¤
model_2/tf.reverse_2/ReverseV2	ReverseV2:model_2/tf.__operators__.getitem_13/strided_slice:output:0,model_2/tf.reverse_2/ReverseV2/axis:output:0*
T0*/
_output_shapes
:         љ
7model_2/tf.__operators__.getitem_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                њ
9model_2/tf.__operators__.getitem_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               њ
9model_2/tf.__operators__.getitem_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Е
1model_2/tf.__operators__.getitem_12/strided_sliceStridedSlice3model_2/batch_normalization_10/FusedBatchNormV3:y:0@model_2/tf.__operators__.getitem_12/strided_slice/stack:output:0Bmodel_2/tf.__operators__.getitem_12/strided_slice/stack_1:output:0Bmodel_2/tf.__operators__.getitem_12/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_maskљ
7model_2/tf.__operators__.getitem_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               њ
9model_2/tf.__operators__.getitem_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"        *       њ
9model_2/tf.__operators__.getitem_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Е
1model_2/tf.__operators__.getitem_14/strided_sliceStridedSlice3model_2/batch_normalization_10/FusedBatchNormV3:y:0@model_2/tf.__operators__.getitem_14/strided_slice/stack:output:0Bmodel_2/tf.__operators__.getitem_14/strided_slice/stack_1:output:0Bmodel_2/tf.__operators__.getitem_14/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_maskљ
7model_2/tf.__operators__.getitem_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                њ
9model_2/tf.__operators__.getitem_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               њ
9model_2/tf.__operators__.getitem_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ю
1model_2/tf.__operators__.getitem_15/strided_sliceStridedSlice'model_2/tf.reverse_2/ReverseV2:output:0@model_2/tf.__operators__.getitem_15/strided_slice/stack:output:0Bmodel_2/tf.__operators__.getitem_15/strided_slice/stack_1:output:0Bmodel_2/tf.__operators__.getitem_15/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_maskљ
7model_2/tf.__operators__.getitem_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"        0       њ
9model_2/tf.__operators__.getitem_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"        3       њ
9model_2/tf.__operators__.getitem_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Е
1model_2/tf.__operators__.getitem_16/strided_sliceStridedSlice3model_2/batch_normalization_10/FusedBatchNormV3:y:0@model_2/tf.__operators__.getitem_16/strided_slice/stack:output:0Bmodel_2/tf.__operators__.getitem_16/strided_slice/stack_1:output:0Bmodel_2/tf.__operators__.getitem_16/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_maskљ
7model_2/tf.__operators__.getitem_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"        	       њ
9model_2/tf.__operators__.getitem_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                њ
9model_2/tf.__operators__.getitem_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ю
1model_2/tf.__operators__.getitem_17/strided_sliceStridedSlice'model_2/tf.reverse_2/ReverseV2:output:0@model_2/tf.__operators__.getitem_17/strided_slice/stack:output:0Bmodel_2/tf.__operators__.getitem_17/strided_slice/stack_1:output:0Bmodel_2/tf.__operators__.getitem_17/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         	*

begin_mask*
end_maska
model_2/tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ч
model_2/tf.concat_2/concatConcatV2:model_2/tf.__operators__.getitem_12/strided_slice:output:0:model_2/tf.__operators__.getitem_13/strided_slice:output:0:model_2/tf.__operators__.getitem_14/strided_slice:output:0:model_2/tf.__operators__.getitem_15/strided_slice:output:0:model_2/tf.__operators__.getitem_16/strided_slice:output:0:model_2/tf.__operators__.getitem_17/strided_slice:output:0(model_2/tf.concat_2/concat/axis:output:0*
N*
T0*/
_output_shapes
:         <н
Amodel_2/spectral_normalization_19/conv2d_17/Conv2D/ReadVariableOpReadVariableOpJmodel_2_spectral_normalization_19_conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ј
2model_2/spectral_normalization_19/conv2d_17/Conv2DConv2D#model_2/tf.concat_2/concat:output:0Imodel_2/spectral_normalization_19/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <*
paddingSAME*
strides
╩
Bmodel_2/spectral_normalization_19/conv2d_17/BiasAdd/ReadVariableOpReadVariableOpKmodel_2_spectral_normalization_19_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ђ
3model_2/spectral_normalization_19/conv2d_17/BiasAddBiasAdd;model_2/spectral_normalization_19/conv2d_17/Conv2D:output:0Jmodel_2/spectral_normalization_19/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <░
0model_2/spectral_normalization_19/conv2d_17/TanhTanh<model_2/spectral_normalization_19/conv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:         <а
-model_2/batch_normalization_11/ReadVariableOpReadVariableOp6model_2_batch_normalization_11_readvariableop_resource*
_output_shapes
:*
dtype0ц
/model_2/batch_normalization_11/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_11_readvariableop_1_resource*
_output_shapes
:*
dtype0┬
>model_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0к
@model_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0 
/model_2/batch_normalization_11/FusedBatchNormV3FusedBatchNormV34model_2/spectral_normalization_19/conv2d_17/Tanh:y:05model_2/batch_normalization_11/ReadVariableOp:value:07model_2/batch_normalization_11/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         <:::::*
epsilon%oЃ:*
is_training( і
IdentityIdentity3model_2/batch_normalization_11/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:         <Щ
NoOpNoOp?^model_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOpA^model_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1.^model_2/batch_normalization_10/ReadVariableOp0^model_2/batch_normalization_10/ReadVariableOp_1?^model_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOpA^model_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1.^model_2/batch_normalization_11/ReadVariableOp0^model_2/batch_normalization_11/ReadVariableOp_1>^model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp@^model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1-^model_2/batch_normalization_8/ReadVariableOp/^model_2/batch_normalization_8/ReadVariableOp_1>^model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp@^model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1-^model_2/batch_normalization_9/ReadVariableOp/^model_2/batch_normalization_9/ReadVariableOp_1C^model_2/spectral_normalization_16/conv2d_14/BiasAdd/ReadVariableOpB^model_2/spectral_normalization_16/conv2d_14/Conv2D/ReadVariableOpC^model_2/spectral_normalization_17/conv2d_15/BiasAdd/ReadVariableOpB^model_2/spectral_normalization_17/conv2d_15/Conv2D/ReadVariableOpC^model_2/spectral_normalization_18/conv2d_16/BiasAdd/ReadVariableOpB^model_2/spectral_normalization_18/conv2d_16/Conv2D/ReadVariableOpC^model_2/spectral_normalization_19/conv2d_17/BiasAdd/ReadVariableOpB^model_2/spectral_normalization_19/conv2d_17/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:         ђ: : : : : : : : : : : : : : : : : : : : : : : : 2ђ
>model_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOp>model_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOp2ё
@model_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1@model_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12^
-model_2/batch_normalization_10/ReadVariableOp-model_2/batch_normalization_10/ReadVariableOp2b
/model_2/batch_normalization_10/ReadVariableOp_1/model_2/batch_normalization_10/ReadVariableOp_12ђ
>model_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOp>model_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOp2ё
@model_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1@model_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12^
-model_2/batch_normalization_11/ReadVariableOp-model_2/batch_normalization_11/ReadVariableOp2b
/model_2/batch_normalization_11/ReadVariableOp_1/model_2/batch_normalization_11/ReadVariableOp_12~
=model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp=model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2ѓ
?model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12\
,model_2/batch_normalization_8/ReadVariableOp,model_2/batch_normalization_8/ReadVariableOp2`
.model_2/batch_normalization_8/ReadVariableOp_1.model_2/batch_normalization_8/ReadVariableOp_12~
=model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp=model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2ѓ
?model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12\
,model_2/batch_normalization_9/ReadVariableOp,model_2/batch_normalization_9/ReadVariableOp2`
.model_2/batch_normalization_9/ReadVariableOp_1.model_2/batch_normalization_9/ReadVariableOp_12ѕ
Bmodel_2/spectral_normalization_16/conv2d_14/BiasAdd/ReadVariableOpBmodel_2/spectral_normalization_16/conv2d_14/BiasAdd/ReadVariableOp2є
Amodel_2/spectral_normalization_16/conv2d_14/Conv2D/ReadVariableOpAmodel_2/spectral_normalization_16/conv2d_14/Conv2D/ReadVariableOp2ѕ
Bmodel_2/spectral_normalization_17/conv2d_15/BiasAdd/ReadVariableOpBmodel_2/spectral_normalization_17/conv2d_15/BiasAdd/ReadVariableOp2є
Amodel_2/spectral_normalization_17/conv2d_15/Conv2D/ReadVariableOpAmodel_2/spectral_normalization_17/conv2d_15/Conv2D/ReadVariableOp2ѕ
Bmodel_2/spectral_normalization_18/conv2d_16/BiasAdd/ReadVariableOpBmodel_2/spectral_normalization_18/conv2d_16/BiasAdd/ReadVariableOp2є
Amodel_2/spectral_normalization_18/conv2d_16/Conv2D/ReadVariableOpAmodel_2/spectral_normalization_18/conv2d_16/Conv2D/ReadVariableOp2ѕ
Bmodel_2/spectral_normalization_19/conv2d_17/BiasAdd/ReadVariableOpBmodel_2/spectral_normalization_19/conv2d_17/BiasAdd/ReadVariableOp2є
Amodel_2/spectral_normalization_19/conv2d_17/Conv2D/ReadVariableOpAmodel_2/spectral_normalization_19/conv2d_17/Conv2D/ReadVariableOp:Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_5
ћ	
м
7__inference_batch_normalization_9_layer_call_fn_2462861

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityѕбStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2460932Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
О
и
V__inference_spectral_normalization_19_layer_call_and_return_conditional_losses_2461233

inputsB
(conv2d_17_conv2d_readvariableop_resource:7
)conv2d_17_biasadd_readvariableop_resource:
identityѕб conv2d_17/BiasAdd/ReadVariableOpбconv2d_17/Conv2D/ReadVariableOpљ
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0┐
conv2d_17/Conv2DConv2Dinputs'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
є
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Г
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           ~
conv2d_17/TanhTanhconv2d_17/BiasAdd:output:0*
T0*A
_output_shapes/
-:+                           {
IdentityIdentityconv2d_17/Tanh:y:0^NoOp*
T0*A
_output_shapes/
-:+                           І
NoOpNoOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
═
Ю
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2460901

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Є
┴
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2462897

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
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
-:+                            н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
═Н
ѕ(
D__inference_model_2_layer_call_and_return_conditional_losses_2462578

inputsT
9spectral_normalization_16_reshape_readvariableop_resource:	ђ@]
Kspectral_normalization_16_spectral_normalize_matmul_readvariableop_resource:@Q
Cspectral_normalization_16_conv2d_14_biasadd_readvariableop_resource:@;
-batch_normalization_8_readvariableop_resource:@=
/batch_normalization_8_readvariableop_1_resource:@L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:@S
9spectral_normalization_17_reshape_readvariableop_resource:@ ]
Kspectral_normalization_17_spectral_normalize_matmul_readvariableop_resource: Q
Cspectral_normalization_17_conv2d_15_biasadd_readvariableop_resource: ;
-batch_normalization_9_readvariableop_resource: =
/batch_normalization_9_readvariableop_1_resource: L
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource: S
9spectral_normalization_18_reshape_readvariableop_resource: ]
Kspectral_normalization_18_spectral_normalize_matmul_readvariableop_resource:Q
Cspectral_normalization_18_conv2d_16_biasadd_readvariableop_resource:<
.batch_normalization_10_readvariableop_resource:>
0batch_normalization_10_readvariableop_1_resource:M
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:S
9spectral_normalization_19_reshape_readvariableop_resource:]
Kspectral_normalization_19_spectral_normalize_matmul_readvariableop_resource:Q
Cspectral_normalization_19_conv2d_17_biasadd_readvariableop_resource:<
.batch_normalization_11_readvariableop_resource:>
0batch_normalization_11_readvariableop_1_resource:M
?batch_normalization_11_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:
identityѕб%batch_normalization_10/AssignNewValueб'batch_normalization_10/AssignNewValue_1б6batch_normalization_10/FusedBatchNormV3/ReadVariableOpб8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_10/ReadVariableOpб'batch_normalization_10/ReadVariableOp_1б%batch_normalization_11/AssignNewValueб'batch_normalization_11/AssignNewValue_1б6batch_normalization_11/FusedBatchNormV3/ReadVariableOpб8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_11/ReadVariableOpб'batch_normalization_11/ReadVariableOp_1б$batch_normalization_8/AssignNewValueб&batch_normalization_8/AssignNewValue_1б5batch_normalization_8/FusedBatchNormV3/ReadVariableOpб7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_8/ReadVariableOpб&batch_normalization_8/ReadVariableOp_1б$batch_normalization_9/AssignNewValueб&batch_normalization_9/AssignNewValue_1б5batch_normalization_9/FusedBatchNormV3/ReadVariableOpб7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_9/ReadVariableOpб&batch_normalization_9/ReadVariableOp_1б0spectral_normalization_16/Reshape/ReadVariableOpб:spectral_normalization_16/conv2d_14/BiasAdd/ReadVariableOpб9spectral_normalization_16/conv2d_14/Conv2D/ReadVariableOpб=spectral_normalization_16/spectral_normalize/AssignVariableOpб?spectral_normalization_16/spectral_normalize/AssignVariableOp_1бBspectral_normalization_16/spectral_normalize/MatMul/ReadVariableOpб;spectral_normalization_16/spectral_normalize/ReadVariableOpб0spectral_normalization_17/Reshape/ReadVariableOpб:spectral_normalization_17/conv2d_15/BiasAdd/ReadVariableOpб9spectral_normalization_17/conv2d_15/Conv2D/ReadVariableOpб=spectral_normalization_17/spectral_normalize/AssignVariableOpб?spectral_normalization_17/spectral_normalize/AssignVariableOp_1бBspectral_normalization_17/spectral_normalize/MatMul/ReadVariableOpб;spectral_normalization_17/spectral_normalize/ReadVariableOpб0spectral_normalization_18/Reshape/ReadVariableOpб:spectral_normalization_18/conv2d_16/BiasAdd/ReadVariableOpб9spectral_normalization_18/conv2d_16/Conv2D/ReadVariableOpб=spectral_normalization_18/spectral_normalize/AssignVariableOpб?spectral_normalization_18/spectral_normalize/AssignVariableOp_1бBspectral_normalization_18/spectral_normalize/MatMul/ReadVariableOpб;spectral_normalization_18/spectral_normalize/ReadVariableOpб0spectral_normalization_19/Reshape/ReadVariableOpб:spectral_normalization_19/conv2d_17/BiasAdd/ReadVariableOpб9spectral_normalization_19/conv2d_17/Conv2D/ReadVariableOpб=spectral_normalization_19/spectral_normalize/AssignVariableOpб?spectral_normalization_19/spectral_normalize/AssignVariableOp_1бBspectral_normalization_19/spectral_normalize/MatMul/ReadVariableOpб;spectral_normalization_19/spectral_normalize/ReadVariableOpE
reshape_2/ShapeShapeinputs*
T0*
_output_shapes
:g
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :\
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :ђ█
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ђ
reshape_2/ReshapeReshapeinputs reshape_2/Reshape/shape:output:0*
T0*0
_output_shapes
:         ђf
up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Ђ
up_sampling2d_6/mulMulup_sampling2d_6/Const:output:0 up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:¤
,up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighborreshape_2/Reshape:output:0up_sampling2d_6/mul:z:0*
T0*0
_output_shapes
:         ђ*
half_pixel_centers(│
0spectral_normalization_16/Reshape/ReadVariableOpReadVariableOp9spectral_normalization_16_reshape_readvariableop_resource*'
_output_shapes
:	ђ@*
dtype0x
'spectral_normalization_16/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   ┬
!spectral_normalization_16/ReshapeReshape8spectral_normalization_16/Reshape/ReadVariableOp:value:00spectral_normalization_16/Reshape/shape:output:0*
T0*
_output_shapes
:	ђ$@╬
Bspectral_normalization_16/spectral_normalize/MatMul/ReadVariableOpReadVariableOpKspectral_normalization_16_spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ы
3spectral_normalization_16/spectral_normalize/MatMulMatMulJspectral_normalization_16/spectral_normalize/MatMul/ReadVariableOp:value:0*spectral_normalization_16/Reshape:output:0*
T0*
_output_shapes
:	ђ$*
transpose_b(│
@spectral_normalization_16/spectral_normalize/l2_normalize/SquareSquare=spectral_normalization_16/spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	ђ$љ
?spectral_normalization_16/spectral_normalize/l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ј
=spectral_normalization_16/spectral_normalize/l2_normalize/SumSumDspectral_normalization_16/spectral_normalize/l2_normalize/Square:y:0Hspectral_normalization_16/spectral_normalize/l2_normalize/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(ѕ
Cspectral_normalization_16/spectral_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+І
Aspectral_normalization_16/spectral_normalize/l2_normalize/MaximumMaximumFspectral_normalization_16/spectral_normalize/l2_normalize/Sum:output:0Lspectral_normalization_16/spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:И
?spectral_normalization_16/spectral_normalize/l2_normalize/RsqrtRsqrtEspectral_normalization_16/spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:Ь
9spectral_normalization_16/spectral_normalize/l2_normalizeMul=spectral_normalization_16/spectral_normalize/MatMul:product:0Cspectral_normalization_16/spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes
:	ђ$М
5spectral_normalization_16/spectral_normalize/MatMul_1MatMul=spectral_normalization_16/spectral_normalize/l2_normalize:z:0*spectral_normalization_16/Reshape:output:0*
T0*
_output_shapes

:@Х
Bspectral_normalization_16/spectral_normalize/l2_normalize_1/SquareSquare?spectral_normalization_16/spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:@њ
Aspectral_normalization_16/spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ћ
?spectral_normalization_16/spectral_normalize/l2_normalize_1/SumSumFspectral_normalization_16/spectral_normalize/l2_normalize_1/Square:y:0Jspectral_normalization_16/spectral_normalize/l2_normalize_1/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(і
Espectral_normalization_16/spectral_normalize/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+Љ
Cspectral_normalization_16/spectral_normalize/l2_normalize_1/MaximumMaximumHspectral_normalization_16/spectral_normalize/l2_normalize_1/Sum:output:0Nspectral_normalization_16/spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:╝
Aspectral_normalization_16/spectral_normalize/l2_normalize_1/RsqrtRsqrtGspectral_normalization_16/spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:з
;spectral_normalization_16/spectral_normalize/l2_normalize_1Mul?spectral_normalization_16/spectral_normalize/MatMul_1:product:0Espectral_normalization_16/spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

:@│
9spectral_normalization_16/spectral_normalize/StopGradientStopGradient?spectral_normalization_16/spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:@┤
;spectral_normalization_16/spectral_normalize/StopGradient_1StopGradient=spectral_normalization_16/spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	ђ$┌
5spectral_normalization_16/spectral_normalize/MatMul_2MatMulDspectral_normalization_16/spectral_normalize/StopGradient_1:output:0*spectral_normalization_16/Reshape:output:0*
T0*
_output_shapes

:@ђ
5spectral_normalization_16/spectral_normalize/MatMul_3MatMul?spectral_normalization_16/spectral_normalize/MatMul_2:product:0Bspectral_normalization_16/spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(Я
=spectral_normalization_16/spectral_normalize/AssignVariableOpAssignVariableOpKspectral_normalization_16_spectral_normalize_matmul_readvariableop_resourceBspectral_normalization_16/spectral_normalize/StopGradient:output:0C^spectral_normalization_16/spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Й
;spectral_normalization_16/spectral_normalize/ReadVariableOpReadVariableOp9spectral_normalization_16_reshape_readvariableop_resource*'
_output_shapes
:	ђ@*
dtype0э
4spectral_normalization_16/spectral_normalize/truedivRealDivCspectral_normalization_16/spectral_normalize/ReadVariableOp:value:0?spectral_normalization_16/spectral_normalize/MatMul_3:product:0*
T0*'
_output_shapes
:	ђ@Њ
:spectral_normalization_16/spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   	   ђ   @   ­
4spectral_normalization_16/spectral_normalize/ReshapeReshape8spectral_normalization_16/spectral_normalize/truediv:z:0Cspectral_normalization_16/spectral_normalize/Reshape/shape:output:0*
T0*'
_output_shapes
:	ђ@э
?spectral_normalization_16/spectral_normalize/AssignVariableOp_1AssignVariableOp9spectral_normalization_16_reshape_readvariableop_resource=spectral_normalization_16/spectral_normalize/Reshape:output:01^spectral_normalization_16/Reshape/ReadVariableOp<^spectral_normalization_16/spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(■
9spectral_normalization_16/conv2d_14/Conv2D/ReadVariableOpReadVariableOp9spectral_normalization_16_reshape_readvariableop_resource@^spectral_normalization_16/spectral_normalize/AssignVariableOp_1*'
_output_shapes
:	ђ@*
dtype0ў
*spectral_normalization_16/conv2d_14/Conv2DConv2D=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0Aspectral_normalization_16/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
║
:spectral_normalization_16/conv2d_14/BiasAdd/ReadVariableOpReadVariableOpCspectral_normalization_16_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ж
+spectral_normalization_16/conv2d_14/BiasAddBiasAdd3spectral_normalization_16/conv2d_14/Conv2D:output:0Bspectral_normalization_16/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @а
(spectral_normalization_16/conv2d_14/ReluRelu4spectral_normalization_16/conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:         @ј
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:@*
dtype0њ
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:@*
dtype0░
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0┤
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Р
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV36spectral_normalization_16/conv2d_14/Relu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<ъ
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(е
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(f
up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Ђ
up_sampling2d_7/mulMulup_sampling2d_7/Const:output:0 up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:я
,up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor*batch_normalization_8/FusedBatchNormV3:y:0up_sampling2d_7/mul:z:0*
T0*/
_output_shapes
:         @*
half_pixel_centers(▓
0spectral_normalization_17/Reshape/ReadVariableOpReadVariableOp9spectral_normalization_17_reshape_readvariableop_resource*&
_output_shapes
:@ *
dtype0x
'spectral_normalization_17/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"        ┬
!spectral_normalization_17/ReshapeReshape8spectral_normalization_17/Reshape/ReadVariableOp:value:00spectral_normalization_17/Reshape/shape:output:0*
T0*
_output_shapes
:	ђ ╬
Bspectral_normalization_17/spectral_normalize/MatMul/ReadVariableOpReadVariableOpKspectral_normalization_17_spectral_normalize_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ы
3spectral_normalization_17/spectral_normalize/MatMulMatMulJspectral_normalization_17/spectral_normalize/MatMul/ReadVariableOp:value:0*spectral_normalization_17/Reshape:output:0*
T0*
_output_shapes
:	ђ*
transpose_b(│
@spectral_normalization_17/spectral_normalize/l2_normalize/SquareSquare=spectral_normalization_17/spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	ђљ
?spectral_normalization_17/spectral_normalize/l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ј
=spectral_normalization_17/spectral_normalize/l2_normalize/SumSumDspectral_normalization_17/spectral_normalize/l2_normalize/Square:y:0Hspectral_normalization_17/spectral_normalize/l2_normalize/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(ѕ
Cspectral_normalization_17/spectral_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+І
Aspectral_normalization_17/spectral_normalize/l2_normalize/MaximumMaximumFspectral_normalization_17/spectral_normalize/l2_normalize/Sum:output:0Lspectral_normalization_17/spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:И
?spectral_normalization_17/spectral_normalize/l2_normalize/RsqrtRsqrtEspectral_normalization_17/spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:Ь
9spectral_normalization_17/spectral_normalize/l2_normalizeMul=spectral_normalization_17/spectral_normalize/MatMul:product:0Cspectral_normalization_17/spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes
:	ђМ
5spectral_normalization_17/spectral_normalize/MatMul_1MatMul=spectral_normalization_17/spectral_normalize/l2_normalize:z:0*spectral_normalization_17/Reshape:output:0*
T0*
_output_shapes

: Х
Bspectral_normalization_17/spectral_normalize/l2_normalize_1/SquareSquare?spectral_normalization_17/spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

: њ
Aspectral_normalization_17/spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ћ
?spectral_normalization_17/spectral_normalize/l2_normalize_1/SumSumFspectral_normalization_17/spectral_normalize/l2_normalize_1/Square:y:0Jspectral_normalization_17/spectral_normalize/l2_normalize_1/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(і
Espectral_normalization_17/spectral_normalize/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+Љ
Cspectral_normalization_17/spectral_normalize/l2_normalize_1/MaximumMaximumHspectral_normalization_17/spectral_normalize/l2_normalize_1/Sum:output:0Nspectral_normalization_17/spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:╝
Aspectral_normalization_17/spectral_normalize/l2_normalize_1/RsqrtRsqrtGspectral_normalization_17/spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:з
;spectral_normalization_17/spectral_normalize/l2_normalize_1Mul?spectral_normalization_17/spectral_normalize/MatMul_1:product:0Espectral_normalization_17/spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

: │
9spectral_normalization_17/spectral_normalize/StopGradientStopGradient?spectral_normalization_17/spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

: ┤
;spectral_normalization_17/spectral_normalize/StopGradient_1StopGradient=spectral_normalization_17/spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	ђ┌
5spectral_normalization_17/spectral_normalize/MatMul_2MatMulDspectral_normalization_17/spectral_normalize/StopGradient_1:output:0*spectral_normalization_17/Reshape:output:0*
T0*
_output_shapes

: ђ
5spectral_normalization_17/spectral_normalize/MatMul_3MatMul?spectral_normalization_17/spectral_normalize/MatMul_2:product:0Bspectral_normalization_17/spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(Я
=spectral_normalization_17/spectral_normalize/AssignVariableOpAssignVariableOpKspectral_normalization_17_spectral_normalize_matmul_readvariableop_resourceBspectral_normalization_17/spectral_normalize/StopGradient:output:0C^spectral_normalization_17/spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(й
;spectral_normalization_17/spectral_normalize/ReadVariableOpReadVariableOp9spectral_normalization_17_reshape_readvariableop_resource*&
_output_shapes
:@ *
dtype0Ш
4spectral_normalization_17/spectral_normalize/truedivRealDivCspectral_normalization_17/spectral_normalize/ReadVariableOp:value:0?spectral_normalization_17/spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:@ Њ
:spectral_normalization_17/spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @       №
4spectral_normalization_17/spectral_normalize/ReshapeReshape8spectral_normalization_17/spectral_normalize/truediv:z:0Cspectral_normalization_17/spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:@ э
?spectral_normalization_17/spectral_normalize/AssignVariableOp_1AssignVariableOp9spectral_normalization_17_reshape_readvariableop_resource=spectral_normalization_17/spectral_normalize/Reshape:output:01^spectral_normalization_17/Reshape/ReadVariableOp<^spectral_normalization_17/spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(§
9spectral_normalization_17/conv2d_15/Conv2D/ReadVariableOpReadVariableOp9spectral_normalization_17_reshape_readvariableop_resource@^spectral_normalization_17/spectral_normalize/AssignVariableOp_1*&
_output_shapes
:@ *
dtype0ў
*spectral_normalization_17/conv2d_15/Conv2DConv2D=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0Aspectral_normalization_17/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
║
:spectral_normalization_17/conv2d_15/BiasAdd/ReadVariableOpReadVariableOpCspectral_normalization_17_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ж
+spectral_normalization_17/conv2d_15/BiasAddBiasAdd3spectral_normalization_17/conv2d_15/Conv2D:output:0Bspectral_normalization_17/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          а
(spectral_normalization_17/conv2d_15/ReluRelu4spectral_normalization_17/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:          ј
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
: *
dtype0њ
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
: *
dtype0░
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0┤
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Р
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV36spectral_normalization_17/conv2d_15/Relu:activations:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
exponential_avg_factor%
О#<ъ
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(е
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(f
up_sampling2d_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Ђ
up_sampling2d_8/mulMulup_sampling2d_8/Const:output:0 up_sampling2d_8/Const_1:output:0*
T0*
_output_shapes
:я
,up_sampling2d_8/resize/ResizeNearestNeighborResizeNearestNeighbor*batch_normalization_9/FusedBatchNormV3:y:0up_sampling2d_8/mul:z:0*
T0*/
_output_shapes
:         < *
half_pixel_centers(▓
0spectral_normalization_18/Reshape/ReadVariableOpReadVariableOp9spectral_normalization_18_reshape_readvariableop_resource*&
_output_shapes
: *
dtype0x
'spectral_normalization_18/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┴
!spectral_normalization_18/ReshapeReshape8spectral_normalization_18/Reshape/ReadVariableOp:value:00spectral_normalization_18/Reshape/shape:output:0*
T0*
_output_shapes

:`╬
Bspectral_normalization_18/spectral_normalize/MatMul/ReadVariableOpReadVariableOpKspectral_normalization_18_spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0ы
3spectral_normalization_18/spectral_normalize/MatMulMatMulJspectral_normalization_18/spectral_normalize/MatMul/ReadVariableOp:value:0*spectral_normalization_18/Reshape:output:0*
T0*
_output_shapes

:`*
transpose_b(▓
@spectral_normalization_18/spectral_normalize/l2_normalize/SquareSquare=spectral_normalization_18/spectral_normalize/MatMul:product:0*
T0*
_output_shapes

:`љ
?spectral_normalization_18/spectral_normalize/l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ј
=spectral_normalization_18/spectral_normalize/l2_normalize/SumSumDspectral_normalization_18/spectral_normalize/l2_normalize/Square:y:0Hspectral_normalization_18/spectral_normalize/l2_normalize/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(ѕ
Cspectral_normalization_18/spectral_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+І
Aspectral_normalization_18/spectral_normalize/l2_normalize/MaximumMaximumFspectral_normalization_18/spectral_normalize/l2_normalize/Sum:output:0Lspectral_normalization_18/spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:И
?spectral_normalization_18/spectral_normalize/l2_normalize/RsqrtRsqrtEspectral_normalization_18/spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:ь
9spectral_normalization_18/spectral_normalize/l2_normalizeMul=spectral_normalization_18/spectral_normalize/MatMul:product:0Cspectral_normalization_18/spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes

:`М
5spectral_normalization_18/spectral_normalize/MatMul_1MatMul=spectral_normalization_18/spectral_normalize/l2_normalize:z:0*spectral_normalization_18/Reshape:output:0*
T0*
_output_shapes

:Х
Bspectral_normalization_18/spectral_normalize/l2_normalize_1/SquareSquare?spectral_normalization_18/spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:њ
Aspectral_normalization_18/spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ћ
?spectral_normalization_18/spectral_normalize/l2_normalize_1/SumSumFspectral_normalization_18/spectral_normalize/l2_normalize_1/Square:y:0Jspectral_normalization_18/spectral_normalize/l2_normalize_1/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(і
Espectral_normalization_18/spectral_normalize/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+Љ
Cspectral_normalization_18/spectral_normalize/l2_normalize_1/MaximumMaximumHspectral_normalization_18/spectral_normalize/l2_normalize_1/Sum:output:0Nspectral_normalization_18/spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:╝
Aspectral_normalization_18/spectral_normalize/l2_normalize_1/RsqrtRsqrtGspectral_normalization_18/spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:з
;spectral_normalization_18/spectral_normalize/l2_normalize_1Mul?spectral_normalization_18/spectral_normalize/MatMul_1:product:0Espectral_normalization_18/spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

:│
9spectral_normalization_18/spectral_normalize/StopGradientStopGradient?spectral_normalization_18/spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:│
;spectral_normalization_18/spectral_normalize/StopGradient_1StopGradient=spectral_normalization_18/spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes

:`┌
5spectral_normalization_18/spectral_normalize/MatMul_2MatMulDspectral_normalization_18/spectral_normalize/StopGradient_1:output:0*spectral_normalization_18/Reshape:output:0*
T0*
_output_shapes

:ђ
5spectral_normalization_18/spectral_normalize/MatMul_3MatMul?spectral_normalization_18/spectral_normalize/MatMul_2:product:0Bspectral_normalization_18/spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(Я
=spectral_normalization_18/spectral_normalize/AssignVariableOpAssignVariableOpKspectral_normalization_18_spectral_normalize_matmul_readvariableop_resourceBspectral_normalization_18/spectral_normalize/StopGradient:output:0C^spectral_normalization_18/spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(й
;spectral_normalization_18/spectral_normalize/ReadVariableOpReadVariableOp9spectral_normalization_18_reshape_readvariableop_resource*&
_output_shapes
: *
dtype0Ш
4spectral_normalization_18/spectral_normalize/truedivRealDivCspectral_normalization_18/spectral_normalize/ReadVariableOp:value:0?spectral_normalization_18/spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
: Њ
:spectral_normalization_18/spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             №
4spectral_normalization_18/spectral_normalize/ReshapeReshape8spectral_normalization_18/spectral_normalize/truediv:z:0Cspectral_normalization_18/spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
: э
?spectral_normalization_18/spectral_normalize/AssignVariableOp_1AssignVariableOp9spectral_normalization_18_reshape_readvariableop_resource=spectral_normalization_18/spectral_normalize/Reshape:output:01^spectral_normalization_18/Reshape/ReadVariableOp<^spectral_normalization_18/spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(§
9spectral_normalization_18/conv2d_16/Conv2D/ReadVariableOpReadVariableOp9spectral_normalization_18_reshape_readvariableop_resource@^spectral_normalization_18/spectral_normalize/AssignVariableOp_1*&
_output_shapes
: *
dtype0ў
*spectral_normalization_18/conv2d_16/Conv2DConv2D=up_sampling2d_8/resize/ResizeNearestNeighbor:resized_images:0Aspectral_normalization_18/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <*
paddingSAME*
strides
║
:spectral_normalization_18/conv2d_16/BiasAdd/ReadVariableOpReadVariableOpCspectral_normalization_18_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ж
+spectral_normalization_18/conv2d_16/BiasAddBiasAdd3spectral_normalization_18/conv2d_16/Conv2D:output:0Bspectral_normalization_18/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <а
(spectral_normalization_18/conv2d_16/ReluRelu4spectral_normalization_18/conv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:         <љ
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
:*
dtype0ћ
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
:*
dtype0▓
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Х
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0у
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV36spectral_normalization_18/conv2d_16/Relu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         <:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<б
%batch_normalization_10/AssignNewValueAssignVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource4batch_normalization_10/FusedBatchNormV3:batch_mean:07^batch_normalization_10/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(г
'batch_normalization_10/AssignNewValue_1AssignVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_10/FusedBatchNormV3:batch_variance:09^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(ѕ
/tf.__operators__.getitem_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ђ
)tf.__operators__.getitem_13/strided_sliceStridedSlice+batch_normalization_10/FusedBatchNormV3:y:08tf.__operators__.getitem_13/strided_slice/stack:output:0:tf.__operators__.getitem_13/strided_slice/stack_1:output:0:tf.__operators__.getitem_13/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_maskl
tf.reverse_2/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB"      и
tf.reverse_2/ReverseV2	ReverseV22tf.__operators__.getitem_13/strided_slice:output:0$tf.reverse_2/ReverseV2/axis:output:0*
T0*/
_output_shapes
:         ѕ
/tf.__operators__.getitem_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                і
1tf.__operators__.getitem_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ђ
)tf.__operators__.getitem_12/strided_sliceStridedSlice+batch_normalization_10/FusedBatchNormV3:y:08tf.__operators__.getitem_12/strided_slice/stack:output:0:tf.__operators__.getitem_12/strided_slice/stack_1:output:0:tf.__operators__.getitem_12/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_maskѕ
/tf.__operators__.getitem_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"        *       і
1tf.__operators__.getitem_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ђ
)tf.__operators__.getitem_14/strided_sliceStridedSlice+batch_normalization_10/FusedBatchNormV3:y:08tf.__operators__.getitem_14/strided_slice/stack:output:0:tf.__operators__.getitem_14/strided_slice/stack_1:output:0:tf.__operators__.getitem_14/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_maskѕ
/tf.__operators__.getitem_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                і
1tf.__operators__.getitem_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ш
)tf.__operators__.getitem_15/strided_sliceStridedSlicetf.reverse_2/ReverseV2:output:08tf.__operators__.getitem_15/strided_slice/stack:output:0:tf.__operators__.getitem_15/strided_slice/stack_1:output:0:tf.__operators__.getitem_15/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_maskѕ
/tf.__operators__.getitem_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"        0       і
1tf.__operators__.getitem_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"        3       і
1tf.__operators__.getitem_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ђ
)tf.__operators__.getitem_16/strided_sliceStridedSlice+batch_normalization_10/FusedBatchNormV3:y:08tf.__operators__.getitem_16/strided_slice/stack:output:0:tf.__operators__.getitem_16/strided_slice/stack_1:output:0:tf.__operators__.getitem_16/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         *

begin_mask*
end_maskѕ
/tf.__operators__.getitem_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"        	       і
1tf.__operators__.getitem_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                і
1tf.__operators__.getitem_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ш
)tf.__operators__.getitem_17/strided_sliceStridedSlicetf.reverse_2/ReverseV2:output:08tf.__operators__.getitem_17/strided_slice/stack:output:0:tf.__operators__.getitem_17/strided_slice/stack_1:output:0:tf.__operators__.getitem_17/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         	*

begin_mask*
end_maskY
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :╗
tf.concat_2/concatConcatV22tf.__operators__.getitem_12/strided_slice:output:02tf.__operators__.getitem_13/strided_slice:output:02tf.__operators__.getitem_14/strided_slice:output:02tf.__operators__.getitem_15/strided_slice:output:02tf.__operators__.getitem_16/strided_slice:output:02tf.__operators__.getitem_17/strided_slice:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*/
_output_shapes
:         <▓
0spectral_normalization_19/Reshape/ReadVariableOpReadVariableOp9spectral_normalization_19_reshape_readvariableop_resource*&
_output_shapes
:*
dtype0x
'spectral_normalization_19/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┴
!spectral_normalization_19/ReshapeReshape8spectral_normalization_19/Reshape/ReadVariableOp:value:00spectral_normalization_19/Reshape/shape:output:0*
T0*
_output_shapes

:╬
Bspectral_normalization_19/spectral_normalize/MatMul/ReadVariableOpReadVariableOpKspectral_normalization_19_spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0ы
3spectral_normalization_19/spectral_normalize/MatMulMatMulJspectral_normalization_19/spectral_normalize/MatMul/ReadVariableOp:value:0*spectral_normalization_19/Reshape:output:0*
T0*
_output_shapes

:*
transpose_b(▓
@spectral_normalization_19/spectral_normalize/l2_normalize/SquareSquare=spectral_normalization_19/spectral_normalize/MatMul:product:0*
T0*
_output_shapes

:љ
?spectral_normalization_19/spectral_normalize/l2_normalize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ј
=spectral_normalization_19/spectral_normalize/l2_normalize/SumSumDspectral_normalization_19/spectral_normalize/l2_normalize/Square:y:0Hspectral_normalization_19/spectral_normalize/l2_normalize/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(ѕ
Cspectral_normalization_19/spectral_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+І
Aspectral_normalization_19/spectral_normalize/l2_normalize/MaximumMaximumFspectral_normalization_19/spectral_normalize/l2_normalize/Sum:output:0Lspectral_normalization_19/spectral_normalize/l2_normalize/Maximum/y:output:0*
T0*
_output_shapes

:И
?spectral_normalization_19/spectral_normalize/l2_normalize/RsqrtRsqrtEspectral_normalization_19/spectral_normalize/l2_normalize/Maximum:z:0*
T0*
_output_shapes

:ь
9spectral_normalization_19/spectral_normalize/l2_normalizeMul=spectral_normalization_19/spectral_normalize/MatMul:product:0Cspectral_normalization_19/spectral_normalize/l2_normalize/Rsqrt:y:0*
T0*
_output_shapes

:М
5spectral_normalization_19/spectral_normalize/MatMul_1MatMul=spectral_normalization_19/spectral_normalize/l2_normalize:z:0*spectral_normalization_19/Reshape:output:0*
T0*
_output_shapes

:Х
Bspectral_normalization_19/spectral_normalize/l2_normalize_1/SquareSquare?spectral_normalization_19/spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:њ
Aspectral_normalization_19/spectral_normalize/l2_normalize_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ћ
?spectral_normalization_19/spectral_normalize/l2_normalize_1/SumSumFspectral_normalization_19/spectral_normalize/l2_normalize_1/Square:y:0Jspectral_normalization_19/spectral_normalize/l2_normalize_1/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(і
Espectral_normalization_19/spectral_normalize/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *╠╝ї+Љ
Cspectral_normalization_19/spectral_normalize/l2_normalize_1/MaximumMaximumHspectral_normalization_19/spectral_normalize/l2_normalize_1/Sum:output:0Nspectral_normalization_19/spectral_normalize/l2_normalize_1/Maximum/y:output:0*
T0*
_output_shapes

:╝
Aspectral_normalization_19/spectral_normalize/l2_normalize_1/RsqrtRsqrtGspectral_normalization_19/spectral_normalize/l2_normalize_1/Maximum:z:0*
T0*
_output_shapes

:з
;spectral_normalization_19/spectral_normalize/l2_normalize_1Mul?spectral_normalization_19/spectral_normalize/MatMul_1:product:0Espectral_normalization_19/spectral_normalize/l2_normalize_1/Rsqrt:y:0*
T0*
_output_shapes

:│
9spectral_normalization_19/spectral_normalize/StopGradientStopGradient?spectral_normalization_19/spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:│
;spectral_normalization_19/spectral_normalize/StopGradient_1StopGradient=spectral_normalization_19/spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes

:┌
5spectral_normalization_19/spectral_normalize/MatMul_2MatMulDspectral_normalization_19/spectral_normalize/StopGradient_1:output:0*spectral_normalization_19/Reshape:output:0*
T0*
_output_shapes

:ђ
5spectral_normalization_19/spectral_normalize/MatMul_3MatMul?spectral_normalization_19/spectral_normalize/MatMul_2:product:0Bspectral_normalization_19/spectral_normalize/StopGradient:output:0*
T0*
_output_shapes

:*
transpose_b(Я
=spectral_normalization_19/spectral_normalize/AssignVariableOpAssignVariableOpKspectral_normalization_19_spectral_normalize_matmul_readvariableop_resourceBspectral_normalization_19/spectral_normalize/StopGradient:output:0C^spectral_normalization_19/spectral_normalize/MatMul/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(й
;spectral_normalization_19/spectral_normalize/ReadVariableOpReadVariableOp9spectral_normalization_19_reshape_readvariableop_resource*&
_output_shapes
:*
dtype0Ш
4spectral_normalization_19/spectral_normalize/truedivRealDivCspectral_normalization_19/spectral_normalize/ReadVariableOp:value:0?spectral_normalization_19/spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:Њ
:spectral_normalization_19/spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            №
4spectral_normalization_19/spectral_normalize/ReshapeReshape8spectral_normalization_19/spectral_normalize/truediv:z:0Cspectral_normalization_19/spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:э
?spectral_normalization_19/spectral_normalize/AssignVariableOp_1AssignVariableOp9spectral_normalization_19_reshape_readvariableop_resource=spectral_normalization_19/spectral_normalize/Reshape:output:01^spectral_normalization_19/Reshape/ReadVariableOp<^spectral_normalization_19/spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(§
9spectral_normalization_19/conv2d_17/Conv2D/ReadVariableOpReadVariableOp9spectral_normalization_19_reshape_readvariableop_resource@^spectral_normalization_19/spectral_normalize/AssignVariableOp_1*&
_output_shapes
:*
dtype0Ш
*spectral_normalization_19/conv2d_17/Conv2DConv2Dtf.concat_2/concat:output:0Aspectral_normalization_19/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <*
paddingSAME*
strides
║
:spectral_normalization_19/conv2d_17/BiasAdd/ReadVariableOpReadVariableOpCspectral_normalization_19_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ж
+spectral_normalization_19/conv2d_17/BiasAddBiasAdd3spectral_normalization_19/conv2d_17/Conv2D:output:0Bspectral_normalization_19/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <а
(spectral_normalization_19/conv2d_17/TanhTanh4spectral_normalization_19/conv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:         <љ
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes
:*
dtype0ћ
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes
:*
dtype0▓
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Х
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0П
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3,spectral_normalization_19/conv2d_17/Tanh:y:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         <:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<б
%batch_normalization_11/AssignNewValueAssignVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource4batch_normalization_11/FusedBatchNormV3:batch_mean:07^batch_normalization_11/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(г
'batch_normalization_11/AssignNewValue_1AssignVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_11/FusedBatchNormV3:batch_variance:09^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(ѓ
IdentityIdentity+batch_normalization_11/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:         <я
NoOpNoOp&^batch_normalization_10/AssignNewValue(^batch_normalization_10/AssignNewValue_17^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_1&^batch_normalization_11/AssignNewValue(^batch_normalization_11/AssignNewValue_17^batch_normalization_11/FusedBatchNormV3/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_11/ReadVariableOp(^batch_normalization_11/ReadVariableOp_1%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_11^spectral_normalization_16/Reshape/ReadVariableOp;^spectral_normalization_16/conv2d_14/BiasAdd/ReadVariableOp:^spectral_normalization_16/conv2d_14/Conv2D/ReadVariableOp>^spectral_normalization_16/spectral_normalize/AssignVariableOp@^spectral_normalization_16/spectral_normalize/AssignVariableOp_1C^spectral_normalization_16/spectral_normalize/MatMul/ReadVariableOp<^spectral_normalization_16/spectral_normalize/ReadVariableOp1^spectral_normalization_17/Reshape/ReadVariableOp;^spectral_normalization_17/conv2d_15/BiasAdd/ReadVariableOp:^spectral_normalization_17/conv2d_15/Conv2D/ReadVariableOp>^spectral_normalization_17/spectral_normalize/AssignVariableOp@^spectral_normalization_17/spectral_normalize/AssignVariableOp_1C^spectral_normalization_17/spectral_normalize/MatMul/ReadVariableOp<^spectral_normalization_17/spectral_normalize/ReadVariableOp1^spectral_normalization_18/Reshape/ReadVariableOp;^spectral_normalization_18/conv2d_16/BiasAdd/ReadVariableOp:^spectral_normalization_18/conv2d_16/Conv2D/ReadVariableOp>^spectral_normalization_18/spectral_normalize/AssignVariableOp@^spectral_normalization_18/spectral_normalize/AssignVariableOp_1C^spectral_normalization_18/spectral_normalize/MatMul/ReadVariableOp<^spectral_normalization_18/spectral_normalize/ReadVariableOp1^spectral_normalization_19/Reshape/ReadVariableOp;^spectral_normalization_19/conv2d_17/BiasAdd/ReadVariableOp:^spectral_normalization_19/conv2d_17/Conv2D/ReadVariableOp>^spectral_normalization_19/spectral_normalize/AssignVariableOp@^spectral_normalization_19/spectral_normalize/AssignVariableOp_1C^spectral_normalization_19/spectral_normalize/MatMul/ReadVariableOp<^spectral_normalization_19/spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:         ђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_10/AssignNewValue%batch_normalization_10/AssignNewValue2R
'batch_normalization_10/AssignNewValue_1'batch_normalization_10/AssignNewValue_12p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12N
%batch_normalization_11/AssignNewValue%batch_normalization_11/AssignNewValue2R
'batch_normalization_11/AssignNewValue_1'batch_normalization_11/AssignNewValue_12p
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp6batch_normalization_11/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_18batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_11/ReadVariableOp%batch_normalization_11/ReadVariableOp2R
'batch_normalization_11/ReadVariableOp_1'batch_normalization_11/ReadVariableOp_12L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12d
0spectral_normalization_16/Reshape/ReadVariableOp0spectral_normalization_16/Reshape/ReadVariableOp2x
:spectral_normalization_16/conv2d_14/BiasAdd/ReadVariableOp:spectral_normalization_16/conv2d_14/BiasAdd/ReadVariableOp2v
9spectral_normalization_16/conv2d_14/Conv2D/ReadVariableOp9spectral_normalization_16/conv2d_14/Conv2D/ReadVariableOp2~
=spectral_normalization_16/spectral_normalize/AssignVariableOp=spectral_normalization_16/spectral_normalize/AssignVariableOp2ѓ
?spectral_normalization_16/spectral_normalize/AssignVariableOp_1?spectral_normalization_16/spectral_normalize/AssignVariableOp_12ѕ
Bspectral_normalization_16/spectral_normalize/MatMul/ReadVariableOpBspectral_normalization_16/spectral_normalize/MatMul/ReadVariableOp2z
;spectral_normalization_16/spectral_normalize/ReadVariableOp;spectral_normalization_16/spectral_normalize/ReadVariableOp2d
0spectral_normalization_17/Reshape/ReadVariableOp0spectral_normalization_17/Reshape/ReadVariableOp2x
:spectral_normalization_17/conv2d_15/BiasAdd/ReadVariableOp:spectral_normalization_17/conv2d_15/BiasAdd/ReadVariableOp2v
9spectral_normalization_17/conv2d_15/Conv2D/ReadVariableOp9spectral_normalization_17/conv2d_15/Conv2D/ReadVariableOp2~
=spectral_normalization_17/spectral_normalize/AssignVariableOp=spectral_normalization_17/spectral_normalize/AssignVariableOp2ѓ
?spectral_normalization_17/spectral_normalize/AssignVariableOp_1?spectral_normalization_17/spectral_normalize/AssignVariableOp_12ѕ
Bspectral_normalization_17/spectral_normalize/MatMul/ReadVariableOpBspectral_normalization_17/spectral_normalize/MatMul/ReadVariableOp2z
;spectral_normalization_17/spectral_normalize/ReadVariableOp;spectral_normalization_17/spectral_normalize/ReadVariableOp2d
0spectral_normalization_18/Reshape/ReadVariableOp0spectral_normalization_18/Reshape/ReadVariableOp2x
:spectral_normalization_18/conv2d_16/BiasAdd/ReadVariableOp:spectral_normalization_18/conv2d_16/BiasAdd/ReadVariableOp2v
9spectral_normalization_18/conv2d_16/Conv2D/ReadVariableOp9spectral_normalization_18/conv2d_16/Conv2D/ReadVariableOp2~
=spectral_normalization_18/spectral_normalize/AssignVariableOp=spectral_normalization_18/spectral_normalize/AssignVariableOp2ѓ
?spectral_normalization_18/spectral_normalize/AssignVariableOp_1?spectral_normalization_18/spectral_normalize/AssignVariableOp_12ѕ
Bspectral_normalization_18/spectral_normalize/MatMul/ReadVariableOpBspectral_normalization_18/spectral_normalize/MatMul/ReadVariableOp2z
;spectral_normalization_18/spectral_normalize/ReadVariableOp;spectral_normalization_18/spectral_normalize/ReadVariableOp2d
0spectral_normalization_19/Reshape/ReadVariableOp0spectral_normalization_19/Reshape/ReadVariableOp2x
:spectral_normalization_19/conv2d_17/BiasAdd/ReadVariableOp:spectral_normalization_19/conv2d_17/BiasAdd/ReadVariableOp2v
9spectral_normalization_19/conv2d_17/Conv2D/ReadVariableOp9spectral_normalization_19/conv2d_17/Conv2D/ReadVariableOp2~
=spectral_normalization_19/spectral_normalize/AssignVariableOp=spectral_normalization_19/spectral_normalize/AssignVariableOp2ѓ
?spectral_normalization_19/spectral_normalize/AssignVariableOp_1?spectral_normalization_19/spectral_normalize/AssignVariableOp_12ѕ
Bspectral_normalization_19/spectral_normalize/MatMul/ReadVariableOpBspectral_normalization_19/spectral_normalize/MatMul/ReadVariableOp2z
;spectral_normalization_19/spectral_normalize/ReadVariableOp;spectral_normalization_19/spectral_normalize/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Р
и
V__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_2462795

inputsB
(conv2d_15_conv2d_readvariableop_resource:@ 7
)conv2d_15_biasadd_readvariableop_resource: 
identityѕб conv2d_15/BiasAdd/ReadVariableOpбconv2d_15/Conv2D/ReadVariableOpљ
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0┐
conv2d_15/Conv2DConv2Dinputs'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
є
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Г
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            ~
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*A
_output_shapes/
-:+                            Ё
IdentityIdentityconv2d_15/Relu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                            І
NoOpNoOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           @: : 2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
я
№
)__inference_model_2_layer_call_fn_2461825
input_5"
unknown:	ђ@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@#
	unknown_6:@ 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: $

unknown_13: 

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:$

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:
identityѕбStatefulPartitionedCall╩
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *.
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_2461705Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:         ђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_5
¤B
ь
 __inference__traced_save_2463287
file_prefix?
;savev2_spectral_normalization_16_kernel_read_readvariableop=
9savev2_spectral_normalization_16_sn_u_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop?
;savev2_spectral_normalization_17_kernel_read_readvariableop=
9savev2_spectral_normalization_17_sn_u_read_readvariableop:
6savev2_batch_normalization_9_gamma_read_readvariableop9
5savev2_batch_normalization_9_beta_read_readvariableop@
<savev2_batch_normalization_9_moving_mean_read_readvariableopD
@savev2_batch_normalization_9_moving_variance_read_readvariableop?
;savev2_spectral_normalization_18_kernel_read_readvariableop=
9savev2_spectral_normalization_18_sn_u_read_readvariableop;
7savev2_batch_normalization_10_gamma_read_readvariableop:
6savev2_batch_normalization_10_beta_read_readvariableopA
=savev2_batch_normalization_10_moving_mean_read_readvariableopE
Asavev2_batch_normalization_10_moving_variance_read_readvariableop?
;savev2_spectral_normalization_19_kernel_read_readvariableop=
9savev2_spectral_normalization_19_sn_u_read_readvariableop;
7savev2_batch_normalization_11_gamma_read_readvariableop:
6savev2_batch_normalization_11_beta_read_readvariableopA
=savev2_batch_normalization_11_moving_mean_read_readvariableopE
Asavev2_batch_normalization_11_moving_variance_read_readvariableop=
9savev2_spectral_normalization_16_bias_read_readvariableop=
9savev2_spectral_normalization_17_bias_read_readvariableop=
9savev2_spectral_normalization_18_bias_read_readvariableop=
9savev2_spectral_normalization_19_bias_read_readvariableop
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
: Ч
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ц
valueЏBўB1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/sn_u/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-2/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/sn_u/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-4/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/sn_u/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-6/w/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/sn_u/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHД
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B ▄
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0;savev2_spectral_normalization_16_kernel_read_readvariableop9savev2_spectral_normalization_16_sn_u_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop;savev2_spectral_normalization_17_kernel_read_readvariableop9savev2_spectral_normalization_17_sn_u_read_readvariableop6savev2_batch_normalization_9_gamma_read_readvariableop5savev2_batch_normalization_9_beta_read_readvariableop<savev2_batch_normalization_9_moving_mean_read_readvariableop@savev2_batch_normalization_9_moving_variance_read_readvariableop;savev2_spectral_normalization_18_kernel_read_readvariableop9savev2_spectral_normalization_18_sn_u_read_readvariableop7savev2_batch_normalization_10_gamma_read_readvariableop6savev2_batch_normalization_10_beta_read_readvariableop=savev2_batch_normalization_10_moving_mean_read_readvariableopAsavev2_batch_normalization_10_moving_variance_read_readvariableop;savev2_spectral_normalization_19_kernel_read_readvariableop9savev2_spectral_normalization_19_sn_u_read_readvariableop7savev2_batch_normalization_11_gamma_read_readvariableop6savev2_batch_normalization_11_beta_read_readvariableop=savev2_batch_normalization_11_moving_mean_read_readvariableopAsavev2_batch_normalization_11_moving_variance_read_readvariableop9savev2_spectral_normalization_16_bias_read_readvariableop9savev2_spectral_normalization_17_bias_read_readvariableop9savev2_spectral_normalization_18_bias_read_readvariableop9savev2_spectral_normalization_19_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *+
dtypes!
2љ
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

identity_1Identity_1:output:0*ѓ
_input_shapes­
ь: :	ђ@:@:@:@:@:@:@ : : : : : : ::::::::::::@: ::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:	ђ@:$ 

_output_shapes

:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@ :$ 

_output_shapes

: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:@: 

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
Х
э
)__inference_model_2_layer_call_fn_2461300
input_5"
unknown:	ђ@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identityѕбStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_2461249Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:         ђ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_5
╬
ъ
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2463162

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
в
з
%__inference_signature_wrapper_2462072
input_5"
unknown:	ђ@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: 

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identityѕбStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         <*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *+
f&R$
"__inference__wrapped_model_2460777w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         <`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:         ђ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_5
ћ
h
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_2462764

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:х
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(ў
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ї8
Е
V__inference_spectral_normalization_18_layer_call_and_return_conditional_losses_2461413

inputs9
reshape_readvariableop_resource: C
1spectral_normalize_matmul_readvariableop_resource:7
)conv2d_16_biasadd_readvariableop_resource:
identityѕбReshape/ReadVariableOpб conv2d_16/BiasAdd/ReadVariableOpбconv2d_16/Conv2D/ReadVariableOpб#spectral_normalize/AssignVariableOpб%spectral_normalize/AssignVariableOp_1б(spectral_normalize/MatMul/ReadVariableOpб!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
: *
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       s
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes

:`џ
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Б
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

:`Ё
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes

:ѓ
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:x
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

:
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes

:`ї
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes

:▓
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
: *
dtype0е
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
: y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             А
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
: ш
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(»
conv2d_16/Conv2D/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
: *
dtype0┐
conv2d_16/Conv2DConv2Dinputs'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
є
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Г
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           ~
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*A
_output_shapes/
-:+                           Ё
IdentityIdentityconv2d_16/Relu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           ┴
NoOpNoOp^Reshape/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                            : : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
ќ	
м
7__inference_batch_normalization_8_layer_call_fn_2462698

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2460818Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ќ	
М
8__inference_batch_normalization_11_layer_call_fn_2463144

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
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
GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2461079Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ѓ8
Е
V__inference_spectral_normalization_19_layer_call_and_return_conditional_losses_2461352

inputs9
reshape_readvariableop_resource:C
1spectral_normalize_matmul_readvariableop_resource:7
)conv2d_17_biasadd_readvariableop_resource:
identityѕбReshape/ReadVariableOpб conv2d_17/BiasAdd/ReadVariableOpбconv2d_17/Conv2D/ReadVariableOpб#spectral_normalize/AssignVariableOpб%spectral_normalize/AssignVariableOp_1б(spectral_normalize/MatMul/ReadVariableOpб!spectral_normalize/ReadVariableOp~
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*&
_output_shapes
:*
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       s
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes

:џ
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Б
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes

:*
transpose_b(~
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes

:v
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

:Ё
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes

:ѓ
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:x
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

:
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes

:ї
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes

:▓
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
:*
dtype0е
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*&
_output_shapes
:y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            А
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*&
_output_shapes
:ш
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(»
conv2d_17/Conv2D/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*&
_output_shapes
:*
dtype0┐
conv2d_17/Conv2DConv2Dinputs'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
є
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Г
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           ~
conv2d_17/TanhTanhconv2d_17/BiasAdd:output:0*
T0*A
_output_shapes/
-:+                           {
IdentityIdentityconv2d_17/Tanh:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ┴
NoOpNoOp^Reshape/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                           : : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Џ8
ф
V__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_2461535

inputs:
reshape_readvariableop_resource:	ђ@C
1spectral_normalize_matmul_readvariableop_resource:@7
)conv2d_14_biasadd_readvariableop_resource:@
identityѕбReshape/ReadVariableOpб conv2d_14/BiasAdd/ReadVariableOpбconv2d_14/Conv2D/ReadVariableOpб#spectral_normalize/AssignVariableOpб%spectral_normalize/AssignVariableOp_1б(spectral_normalize/MatMul/ReadVariableOpб!spectral_normalize/ReadVariableOp
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*'
_output_shapes
:	ђ@*
dtype0^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   t
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*
_output_shapes
:	ђ$@џ
(spectral_normalize/MatMul/ReadVariableOpReadVariableOp1spectral_normalize_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0ц
spectral_normalize/MatMulMatMul0spectral_normalize/MatMul/ReadVariableOp:value:0Reshape:output:0*
T0*
_output_shapes
:	ђ$*
transpose_b(
&spectral_normalize/l2_normalize/SquareSquare#spectral_normalize/MatMul:product:0*
T0*
_output_shapes
:	ђ$v
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
:	ђ$Ё
spectral_normalize/MatMul_1MatMul#spectral_normalize/l2_normalize:z:0Reshape:output:0*
T0*
_output_shapes

:@ѓ
(spectral_normalize/l2_normalize_1/SquareSquare%spectral_normalize/MatMul_1:product:0*
T0*
_output_shapes

:@x
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

:@
spectral_normalize/StopGradientStopGradient%spectral_normalize/l2_normalize_1:z:0*
T0*
_output_shapes

:@ђ
!spectral_normalize/StopGradient_1StopGradient#spectral_normalize/l2_normalize:z:0*
T0*
_output_shapes
:	ђ$ї
spectral_normalize/MatMul_2MatMul*spectral_normalize/StopGradient_1:output:0Reshape:output:0*
T0*
_output_shapes

:@▓
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
:	ђ@*
dtype0Е
spectral_normalize/truedivRealDiv)spectral_normalize/ReadVariableOp:value:0%spectral_normalize/MatMul_3:product:0*
T0*'
_output_shapes
:	ђ@y
 spectral_normalize/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   	   ђ   @   б
spectral_normalize/ReshapeReshapespectral_normalize/truediv:z:0)spectral_normalize/Reshape/shape:output:0*
T0*'
_output_shapes
:	ђ@ш
%spectral_normalize/AssignVariableOp_1AssignVariableOpreshape_readvariableop_resource#spectral_normalize/Reshape:output:0^Reshape/ReadVariableOp"^spectral_normalize/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(░
conv2d_14/Conv2D/ReadVariableOpReadVariableOpreshape_readvariableop_resource&^spectral_normalize/AssignVariableOp_1*'
_output_shapes
:	ђ@*
dtype0┐
conv2d_14/Conv2DConv2Dinputs'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
є
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Г
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @~
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @Ё
IdentityIdentityconv2d_14/Relu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           @┴
NoOpNoOp^Reshape/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp$^spectral_normalize/AssignVariableOp&^spectral_normalize/AssignVariableOp_1)^spectral_normalize/MatMul/ReadVariableOp"^spectral_normalize/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4:,                           ђ: : : 20
Reshape/ReadVariableOpReshape/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2J
#spectral_normalize/AssignVariableOp#spectral_normalize/AssignVariableOp2N
%spectral_normalize/AssignVariableOp_1%spectral_normalize/AssignVariableOp_12T
(spectral_normalize/MatMul/ReadVariableOp(spectral_normalize/MatMul/ReadVariableOp2F
!spectral_normalize/ReadVariableOp!spectral_normalize/ReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
▒x
┌
D__inference_model_2_layer_call_and_return_conditional_losses_2461705

inputs<
!spectral_normalization_16_2461610:	ђ@3
!spectral_normalization_16_2461612:@/
!spectral_normalization_16_2461614:@+
batch_normalization_8_2461617:@+
batch_normalization_8_2461619:@+
batch_normalization_8_2461621:@+
batch_normalization_8_2461623:@;
!spectral_normalization_17_2461627:@ 3
!spectral_normalization_17_2461629: /
!spectral_normalization_17_2461631: +
batch_normalization_9_2461634: +
batch_normalization_9_2461636: +
batch_normalization_9_2461638: +
batch_normalization_9_2461640: ;
!spectral_normalization_18_2461644: 3
!spectral_normalization_18_2461646:/
!spectral_normalization_18_2461648:,
batch_normalization_10_2461651:,
batch_normalization_10_2461653:,
batch_normalization_10_2461655:,
batch_normalization_10_2461657:;
!spectral_normalization_19_2461688:3
!spectral_normalization_19_2461690:/
!spectral_normalization_19_2461692:,
batch_normalization_11_2461695:,
batch_normalization_11_2461697:,
batch_normalization_11_2461699:,
batch_normalization_11_2461701:
identityѕб.batch_normalization_10/StatefulPartitionedCallб.batch_normalization_11/StatefulPartitionedCallб-batch_normalization_8/StatefulPartitionedCallб-batch_normalization_9/StatefulPartitionedCallб1spectral_normalization_16/StatefulPartitionedCallб1spectral_normalization_17/StatefulPartitionedCallб1spectral_normalization_18/StatefulPartitionedCallб1spectral_normalization_19/StatefulPartitionedCallК
reshape_2/PartitionedCallPartitionedCallinputs*
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
GPU2*0J 8ѓ *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_2461111Ђ
up_sampling2d_6/PartitionedCallPartitionedCall"reshape_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_2460793Ў
1spectral_normalization_16/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_6/PartitionedCall:output:0!spectral_normalization_16_2461610!spectral_normalization_16_2461612!spectral_normalization_16_2461614*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *_
fZRX
V__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_2461535И
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_16/StatefulPartitionedCall:output:0batch_normalization_8_2461617batch_normalization_8_2461619batch_normalization_8_2461621batch_normalization_8_2461623*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2460849ћ
up_sampling2d_7/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_2460876Ў
1spectral_normalization_17/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_7/PartitionedCall:output:0!spectral_normalization_17_2461627!spectral_normalization_17_2461629!spectral_normalization_17_2461631*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *_
fZRX
V__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_2461474И
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_17/StatefulPartitionedCall:output:0batch_normalization_9_2461634batch_normalization_9_2461636batch_normalization_9_2461638batch_normalization_9_2461640*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2460932ћ
up_sampling2d_8/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_2460959Ў
1spectral_normalization_18/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_8/PartitionedCall:output:0!spectral_normalization_18_2461644!spectral_normalization_18_2461646!spectral_normalization_18_2461648*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *_
fZRX
V__inference_spectral_normalization_18_layer_call_and_return_conditional_losses_2461413Й
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_18/StatefulPartitionedCall:output:0batch_normalization_10_2461651batch_normalization_10_2461653batch_normalization_10_2461655batch_normalization_10_2461657*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2461015ѕ
/tf.__operators__.getitem_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ъ
)tf.__operators__.getitem_13/strided_sliceStridedSlice7batch_normalization_10/StatefulPartitionedCall:output:08tf.__operators__.getitem_13/strided_slice/stack:output:0:tf.__operators__.getitem_13/strided_slice/stack_1:output:0:tf.__operators__.getitem_13/strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+                           *

begin_mask*
end_maskl
tf.reverse_2/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB"      ╔
tf.reverse_2/ReverseV2	ReverseV22tf.__operators__.getitem_13/strided_slice:output:0$tf.reverse_2/ReverseV2/axis:output:0*
T0*A
_output_shapes/
-:+                           ѕ
/tf.__operators__.getitem_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                і
1tf.__operators__.getitem_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ъ
)tf.__operators__.getitem_12/strided_sliceStridedSlice7batch_normalization_10/StatefulPartitionedCall:output:08tf.__operators__.getitem_12/strided_slice/stack:output:0:tf.__operators__.getitem_12/strided_slice/stack_1:output:0:tf.__operators__.getitem_12/strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+                           *

begin_mask*
end_maskѕ
/tf.__operators__.getitem_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"        *       і
1tf.__operators__.getitem_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ъ
)tf.__operators__.getitem_14/strided_sliceStridedSlice7batch_normalization_10/StatefulPartitionedCall:output:08tf.__operators__.getitem_14/strided_slice/stack:output:0:tf.__operators__.getitem_14/strided_slice/stack_1:output:0:tf.__operators__.getitem_14/strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+                           *

begin_mask*
end_maskѕ
/tf.__operators__.getitem_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                і
1tf.__operators__.getitem_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Є
)tf.__operators__.getitem_15/strided_sliceStridedSlicetf.reverse_2/ReverseV2:output:08tf.__operators__.getitem_15/strided_slice/stack:output:0:tf.__operators__.getitem_15/strided_slice/stack_1:output:0:tf.__operators__.getitem_15/strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+                           *

begin_mask*
end_maskѕ
/tf.__operators__.getitem_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"        0       і
1tf.__operators__.getitem_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"        3       і
1tf.__operators__.getitem_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ъ
)tf.__operators__.getitem_16/strided_sliceStridedSlice7batch_normalization_10/StatefulPartitionedCall:output:08tf.__operators__.getitem_16/strided_slice/stack:output:0:tf.__operators__.getitem_16/strided_slice/stack_1:output:0:tf.__operators__.getitem_16/strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+                           *

begin_mask*
end_maskѕ
/tf.__operators__.getitem_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"        	       і
1tf.__operators__.getitem_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                і
1tf.__operators__.getitem_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Є
)tf.__operators__.getitem_17/strided_sliceStridedSlicetf.reverse_2/ReverseV2:output:08tf.__operators__.getitem_17/strided_slice/stack:output:0:tf.__operators__.getitem_17/strided_slice/stack_1:output:0:tf.__operators__.getitem_17/strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+                           *

begin_mask*
end_maskY
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :═
tf.concat_2/concatConcatV22tf.__operators__.getitem_12/strided_slice:output:02tf.__operators__.getitem_13/strided_slice:output:02tf.__operators__.getitem_14/strided_slice:output:02tf.__operators__.getitem_15/strided_slice:output:02tf.__operators__.getitem_16/strided_slice:output:02tf.__operators__.getitem_17/strided_slice:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+                           ї
1spectral_normalization_19/StatefulPartitionedCallStatefulPartitionedCalltf.concat_2/concat:output:0!spectral_normalization_19_2461688!spectral_normalization_19_2461690!spectral_normalization_19_2461692*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *_
fZRX
V__inference_spectral_normalization_19_layer_call_and_return_conditional_losses_2461352Й
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_19/StatefulPartitionedCall:output:0batch_normalization_11_2461695batch_normalization_11_2461697batch_normalization_11_2461699batch_normalization_11_2461701*
Tin	
2*
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
GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2461079а
IdentityIdentity7batch_normalization_11/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           п
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall2^spectral_normalization_16/StatefulPartitionedCall2^spectral_normalization_17/StatefulPartitionedCall2^spectral_normalization_18/StatefulPartitionedCall2^spectral_normalization_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:         ђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2f
1spectral_normalization_16/StatefulPartitionedCall1spectral_normalization_16/StatefulPartitionedCall2f
1spectral_normalization_17/StatefulPartitionedCall1spectral_normalization_17/StatefulPartitionedCall2f
1spectral_normalization_18/StatefulPartitionedCall1spectral_normalization_18/StatefulPartitionedCall2f
1spectral_normalization_19/StatefulPartitionedCall1spectral_normalization_19/StatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ѕ
┬
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2461015

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
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
-:+                           н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
┤x
█
D__inference_model_2_layer_call_and_return_conditional_losses_2462017
input_5<
!spectral_normalization_16_2461922:	ђ@3
!spectral_normalization_16_2461924:@/
!spectral_normalization_16_2461926:@+
batch_normalization_8_2461929:@+
batch_normalization_8_2461931:@+
batch_normalization_8_2461933:@+
batch_normalization_8_2461935:@;
!spectral_normalization_17_2461939:@ 3
!spectral_normalization_17_2461941: /
!spectral_normalization_17_2461943: +
batch_normalization_9_2461946: +
batch_normalization_9_2461948: +
batch_normalization_9_2461950: +
batch_normalization_9_2461952: ;
!spectral_normalization_18_2461956: 3
!spectral_normalization_18_2461958:/
!spectral_normalization_18_2461960:,
batch_normalization_10_2461963:,
batch_normalization_10_2461965:,
batch_normalization_10_2461967:,
batch_normalization_10_2461969:;
!spectral_normalization_19_2462000:3
!spectral_normalization_19_2462002:/
!spectral_normalization_19_2462004:,
batch_normalization_11_2462007:,
batch_normalization_11_2462009:,
batch_normalization_11_2462011:,
batch_normalization_11_2462013:
identityѕб.batch_normalization_10/StatefulPartitionedCallб.batch_normalization_11/StatefulPartitionedCallб-batch_normalization_8/StatefulPartitionedCallб-batch_normalization_9/StatefulPartitionedCallб1spectral_normalization_16/StatefulPartitionedCallб1spectral_normalization_17/StatefulPartitionedCallб1spectral_normalization_18/StatefulPartitionedCallб1spectral_normalization_19/StatefulPartitionedCall╚
reshape_2/PartitionedCallPartitionedCallinput_5*
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
GPU2*0J 8ѓ *O
fJRH
F__inference_reshape_2_layer_call_and_return_conditional_losses_2461111Ђ
up_sampling2d_6/PartitionedCallPartitionedCall"reshape_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_2460793Ў
1spectral_normalization_16/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_6/PartitionedCall:output:0!spectral_normalization_16_2461922!spectral_normalization_16_2461924!spectral_normalization_16_2461926*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *_
fZRX
V__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_2461535И
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_16/StatefulPartitionedCall:output:0batch_normalization_8_2461929batch_normalization_8_2461931batch_normalization_8_2461933batch_normalization_8_2461935*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2460849ћ
up_sampling2d_7/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_2460876Ў
1spectral_normalization_17/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_7/PartitionedCall:output:0!spectral_normalization_17_2461939!spectral_normalization_17_2461941!spectral_normalization_17_2461943*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *_
fZRX
V__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_2461474И
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_17/StatefulPartitionedCall:output:0batch_normalization_9_2461946batch_normalization_9_2461948batch_normalization_9_2461950batch_normalization_9_2461952*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2460932ћ
up_sampling2d_8/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_2460959Ў
1spectral_normalization_18/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_8/PartitionedCall:output:0!spectral_normalization_18_2461956!spectral_normalization_18_2461958!spectral_normalization_18_2461960*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *_
fZRX
V__inference_spectral_normalization_18_layer_call_and_return_conditional_losses_2461413Й
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_18/StatefulPartitionedCall:output:0batch_normalization_10_2461963batch_normalization_10_2461965batch_normalization_10_2461967batch_normalization_10_2461969*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2461015ѕ
/tf.__operators__.getitem_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ъ
)tf.__operators__.getitem_13/strided_sliceStridedSlice7batch_normalization_10/StatefulPartitionedCall:output:08tf.__operators__.getitem_13/strided_slice/stack:output:0:tf.__operators__.getitem_13/strided_slice/stack_1:output:0:tf.__operators__.getitem_13/strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+                           *

begin_mask*
end_maskl
tf.reverse_2/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB"      ╔
tf.reverse_2/ReverseV2	ReverseV22tf.__operators__.getitem_13/strided_slice:output:0$tf.reverse_2/ReverseV2/axis:output:0*
T0*A
_output_shapes/
-:+                           ѕ
/tf.__operators__.getitem_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                і
1tf.__operators__.getitem_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ъ
)tf.__operators__.getitem_12/strided_sliceStridedSlice7batch_normalization_10/StatefulPartitionedCall:output:08tf.__operators__.getitem_12/strided_slice/stack:output:0:tf.__operators__.getitem_12/strided_slice/stack_1:output:0:tf.__operators__.getitem_12/strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+                           *

begin_mask*
end_maskѕ
/tf.__operators__.getitem_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"        *       і
1tf.__operators__.getitem_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ъ
)tf.__operators__.getitem_14/strided_sliceStridedSlice7batch_normalization_10/StatefulPartitionedCall:output:08tf.__operators__.getitem_14/strided_slice/stack:output:0:tf.__operators__.getitem_14/strided_slice/stack_1:output:0:tf.__operators__.getitem_14/strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+                           *

begin_mask*
end_maskѕ
/tf.__operators__.getitem_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                і
1tf.__operators__.getitem_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               і
1tf.__operators__.getitem_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Є
)tf.__operators__.getitem_15/strided_sliceStridedSlicetf.reverse_2/ReverseV2:output:08tf.__operators__.getitem_15/strided_slice/stack:output:0:tf.__operators__.getitem_15/strided_slice/stack_1:output:0:tf.__operators__.getitem_15/strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+                           *

begin_mask*
end_maskѕ
/tf.__operators__.getitem_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"        0       і
1tf.__operators__.getitem_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"        3       і
1tf.__operators__.getitem_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Ъ
)tf.__operators__.getitem_16/strided_sliceStridedSlice7batch_normalization_10/StatefulPartitionedCall:output:08tf.__operators__.getitem_16/strided_slice/stack:output:0:tf.__operators__.getitem_16/strided_slice/stack_1:output:0:tf.__operators__.getitem_16/strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+                           *

begin_mask*
end_maskѕ
/tf.__operators__.getitem_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"        	       і
1tf.__operators__.getitem_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                і
1tf.__operators__.getitem_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Є
)tf.__operators__.getitem_17/strided_sliceStridedSlicetf.reverse_2/ReverseV2:output:08tf.__operators__.getitem_17/strided_slice/stack:output:0:tf.__operators__.getitem_17/strided_slice/stack_1:output:0:tf.__operators__.getitem_17/strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+                           *

begin_mask*
end_maskY
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :═
tf.concat_2/concatConcatV22tf.__operators__.getitem_12/strided_slice:output:02tf.__operators__.getitem_13/strided_slice:output:02tf.__operators__.getitem_14/strided_slice:output:02tf.__operators__.getitem_15/strided_slice:output:02tf.__operators__.getitem_16/strided_slice:output:02tf.__operators__.getitem_17/strided_slice:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+                           ї
1spectral_normalization_19/StatefulPartitionedCallStatefulPartitionedCalltf.concat_2/concat:output:0!spectral_normalization_19_2462000!spectral_normalization_19_2462002!spectral_normalization_19_2462004*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *_
fZRX
V__inference_spectral_normalization_19_layer_call_and_return_conditional_losses_2461352Й
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall:spectral_normalization_19/StatefulPartitionedCall:output:0batch_normalization_11_2462007batch_normalization_11_2462009batch_normalization_11_2462011batch_normalization_11_2462013*
Tin	
2*
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
GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2461079а
IdentityIdentity7batch_normalization_11/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           п
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall2^spectral_normalization_16/StatefulPartitionedCall2^spectral_normalization_17/StatefulPartitionedCall2^spectral_normalization_18/StatefulPartitionedCall2^spectral_normalization_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:         ђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2f
1spectral_normalization_16/StatefulPartitionedCall1spectral_normalization_16/StatefulPartitionedCall2f
1spectral_normalization_17/StatefulPartitionedCall1spectral_normalization_17/StatefulPartitionedCall2f
1spectral_normalization_18/StatefulPartitionedCall1spectral_normalization_18/StatefulPartitionedCall2f
1spectral_normalization_19/StatefulPartitionedCall1spectral_normalization_19/StatefulPartitionedCall:Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_5
█
Ь
)__inference_model_2_layer_call_fn_2462186

inputs"
unknown:	ђ@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@#
	unknown_6:@ 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: $

unknown_13: 

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:$

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:
identityѕбStatefulPartitionedCall╔
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
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *.
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_2461705Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:         ђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ћ
h
L__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_2462914

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:х
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(ў
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ў	
М
8__inference_batch_normalization_11_layer_call_fn_2463131

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2461048Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
┌
░
;__inference_spectral_normalization_18_layer_call_fn_2462923

inputs!
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *_
fZRX
V__inference_spectral_normalization_18_layer_call_and_return_conditional_losses_2461179Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                            : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Р
и
V__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_2461152

inputsB
(conv2d_15_conv2d_readvariableop_resource:@ 7
)conv2d_15_biasadd_readvariableop_resource: 
identityѕб conv2d_15/BiasAdd/ReadVariableOpбconv2d_15/Conv2D/ReadVariableOpљ
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0┐
conv2d_15/Conv2DConv2Dinputs'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
є
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Г
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            ~
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*A
_output_shapes/
-:+                            Ё
IdentityIdentityconv2d_15/Relu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                            І
NoOpNoOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           @: : 2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ё	
═
;__inference_spectral_normalization_18_layer_call_fn_2462934

inputs!
unknown: 
	unknown_0:
	unknown_1:
identityѕбStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *_
fZRX
V__inference_spectral_normalization_18_layer_call_and_return_conditional_losses_2461413Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                            : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╬
ъ
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2460984

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
┌
░
;__inference_spectral_normalization_17_layer_call_fn_2462773

inputs!
unknown:@ 
	unknown_0: 
identityѕбStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *_
fZRX
V__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_2461152Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           @: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ѕ
┬
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2463047

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
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
-:+                           н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
═
Ю
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2462879

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs"х	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*┬
serving_default«
<
input_51
serving_default_input_5:0         ђR
batch_normalization_118
StatefulPartitionedCall:0         <tensorflow/serving/predict:гЪ
└
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer_with_weights-6
layer-19
layer_with_weights-7
layer-20
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
Ц
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
Ц
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layer
Н
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
	0layer
1w
2w_shape
3sn_u
3u"
_tf_keras_layer
Ж
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses
:axis
	;gamma
<beta
=moving_mean
>moving_variance"
_tf_keras_layer
Ц
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
Н
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses
	Klayer
Lw
Mw_shape
Nsn_u
Nu"
_tf_keras_layer
Ж
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
Uaxis
	Vgamma
Wbeta
Xmoving_mean
Ymoving_variance"
_tf_keras_layer
Ц
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses"
_tf_keras_layer
Н
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses
	flayer
gw
hw_shape
isn_u
iu"
_tf_keras_layer
Ж
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses
paxis
	qgamma
rbeta
smoving_mean
tmoving_variance"
_tf_keras_layer
(
u	keras_api"
_tf_keras_layer
(
v	keras_api"
_tf_keras_layer
(
w	keras_api"
_tf_keras_layer
(
x	keras_api"
_tf_keras_layer
(
y	keras_api"
_tf_keras_layer
(
z	keras_api"
_tf_keras_layer
(
{	keras_api"
_tf_keras_layer
(
|	keras_api"
_tf_keras_layer
П
}	variables
~trainable_variables
regularization_losses
ђ	keras_api
Ђ__call__
+ѓ&call_and_return_all_conditional_losses

Ѓlayer
ёw
Ёw_shape
	єsn_u
єu"
_tf_keras_layer
ш
Є	variables
ѕtrainable_variables
Ѕregularization_losses
і	keras_api
І__call__
+ї&call_and_return_all_conditional_losses
	Їaxis

јgamma
	Јbeta
љmoving_mean
Љmoving_variance"
_tf_keras_layer
ђ
10
њ1
32
;3
<4
=5
>6
L7
Њ8
N9
V10
W11
X12
Y13
g14
ћ15
i16
q17
r18
s19
t20
ё21
Ћ22
є23
ј24
Ј25
љ26
Љ27"
trackable_list_wrapper
Ю
10
њ1
;2
<3
L4
Њ5
V6
W7
g8
ћ9
q10
r11
ё12
Ћ13
ј14
Ј15"
trackable_list_wrapper
 "
trackable_list_wrapper
¤
ќnon_trainable_variables
Ќlayers
ўmetrics
 Ўlayer_regularization_losses
џlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
р
Џtrace_0
юtrace_1
Юtrace_2
ъtrace_32Ь
)__inference_model_2_layer_call_fn_2461300
)__inference_model_2_layer_call_fn_2462125
)__inference_model_2_layer_call_fn_2462186
)__inference_model_2_layer_call_fn_2461825┐
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
 zЏtrace_0zюtrace_1zЮtrace_2zъtrace_3
═
Ъtrace_0
аtrace_1
Аtrace_2
бtrace_32┌
D__inference_model_2_layer_call_and_return_conditional_losses_2462324
D__inference_model_2_layer_call_and_return_conditional_losses_2462578
D__inference_model_2_layer_call_and_return_conditional_losses_2461917
D__inference_model_2_layer_call_and_return_conditional_losses_2462017┐
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
 zЪtrace_0zаtrace_1zАtrace_2zбtrace_3
═B╩
"__inference__wrapped_model_2460777input_5"ў
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
Бserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
цnon_trainable_variables
Цlayers
дmetrics
 Дlayer_regularization_losses
еlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
ы
Еtrace_02м
+__inference_reshape_2_layer_call_fn_2462583б
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
 zЕtrace_0
ї
фtrace_02ь
F__inference_reshape_2_layer_call_and_return_conditional_losses_2462597б
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
 zфtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Фnon_trainable_variables
гlayers
Гmetrics
 «layer_regularization_losses
»layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
э
░trace_02п
1__inference_up_sampling2d_6_layer_call_fn_2462602б
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
 z░trace_0
њ
▒trace_02з
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_2462614б
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
 z▒trace_0
6
10
њ1
32"
trackable_list_wrapper
/
10
њ1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
▓non_trainable_variables
│layers
┤metrics
 хlayer_regularization_losses
Хlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
в
иtrace_0
Иtrace_12░
;__inference_spectral_normalization_16_layer_call_fn_2462623
;__inference_spectral_normalization_16_layer_call_fn_2462634│
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
 zиtrace_0zИtrace_1
А
╣trace_0
║trace_12Т
V__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_2462645
V__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_2462685│
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
 z╣trace_0z║trace_1
т
╗	variables
╝trainable_variables
йregularization_losses
Й	keras_api
┐__call__
+└&call_and_return_all_conditional_losses

1kernel
	њbias
!┴_jit_compiled_convolution_op"
_tf_keras_layer
;:9	ђ@2 spectral_normalization_16/kernel
 "
trackable_list_wrapper
.:,@2spectral_normalization_16/sn_u
<
;0
<1
=2
>3"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
┬non_trainable_variables
├layers
─metrics
 ┼layer_regularization_losses
кlayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
с
Кtrace_0
╚trace_12е
7__inference_batch_normalization_8_layer_call_fn_2462698
7__inference_batch_normalization_8_layer_call_fn_2462711│
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
 zКtrace_0z╚trace_1
Ў
╔trace_0
╩trace_12я
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2462729
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2462747│
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
 z╔trace_0z╩trace_1
 "
trackable_list_wrapper
):'@2batch_normalization_8/gamma
(:&@2batch_normalization_8/beta
1:/@ (2!batch_normalization_8/moving_mean
5:3@ (2%batch_normalization_8/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╦non_trainable_variables
╠layers
═metrics
 ╬layer_regularization_losses
¤layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
э
лtrace_02п
1__inference_up_sampling2d_7_layer_call_fn_2462752б
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
 zлtrace_0
њ
Лtrace_02з
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_2462764б
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
 zЛtrace_0
6
L0
Њ1
N2"
trackable_list_wrapper
/
L0
Њ1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
мnon_trainable_variables
Мlayers
нmetrics
 Нlayer_regularization_losses
оlayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
в
Оtrace_0
пtrace_12░
;__inference_spectral_normalization_17_layer_call_fn_2462773
;__inference_spectral_normalization_17_layer_call_fn_2462784│
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
 zОtrace_0zпtrace_1
А
┘trace_0
┌trace_12Т
V__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_2462795
V__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_2462835│
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
 z┘trace_0z┌trace_1
т
█	variables
▄trainable_variables
Пregularization_losses
я	keras_api
▀__call__
+Я&call_and_return_all_conditional_losses

Lkernel
	Њbias
!р_jit_compiled_convolution_op"
_tf_keras_layer
::8@ 2 spectral_normalization_17/kernel
 "
trackable_list_wrapper
.:, 2spectral_normalization_17/sn_u
<
V0
W1
X2
Y3"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Рnon_trainable_variables
сlayers
Сmetrics
 тlayer_regularization_losses
Тlayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
с
уtrace_0
Уtrace_12е
7__inference_batch_normalization_9_layer_call_fn_2462848
7__inference_batch_normalization_9_layer_call_fn_2462861│
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
 zуtrace_0zУtrace_1
Ў
жtrace_0
Жtrace_12я
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2462879
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2462897│
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
 zжtrace_0zЖtrace_1
 "
trackable_list_wrapper
):' 2batch_normalization_9/gamma
(:& 2batch_normalization_9/beta
1:/  (2!batch_normalization_9/moving_mean
5:3  (2%batch_normalization_9/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
вnon_trainable_variables
Вlayers
ьmetrics
 Ьlayer_regularization_losses
№layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
э
­trace_02п
1__inference_up_sampling2d_8_layer_call_fn_2462902б
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
 z­trace_0
њ
ыtrace_02з
L__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_2462914б
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
 zыtrace_0
6
g0
ћ1
i2"
trackable_list_wrapper
/
g0
ћ1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Ыnon_trainable_variables
зlayers
Зmetrics
 шlayer_regularization_losses
Шlayer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
в
эtrace_0
Эtrace_12░
;__inference_spectral_normalization_18_layer_call_fn_2462923
;__inference_spectral_normalization_18_layer_call_fn_2462934│
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
 zэtrace_0zЭtrace_1
А
щtrace_0
Щtrace_12Т
V__inference_spectral_normalization_18_layer_call_and_return_conditional_losses_2462945
V__inference_spectral_normalization_18_layer_call_and_return_conditional_losses_2462985│
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
 zщtrace_0zЩtrace_1
т
ч	variables
Чtrainable_variables
§regularization_losses
■	keras_api
 __call__
+ђ&call_and_return_all_conditional_losses

gkernel
	ћbias
!Ђ_jit_compiled_convolution_op"
_tf_keras_layer
::8 2 spectral_normalization_18/kernel
 "
trackable_list_wrapper
.:,2spectral_normalization_18/sn_u
<
q0
r1
s2
t3"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ѓnon_trainable_variables
Ѓlayers
ёmetrics
 Ёlayer_regularization_losses
єlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
т
Єtrace_0
ѕtrace_12ф
8__inference_batch_normalization_10_layer_call_fn_2462998
8__inference_batch_normalization_10_layer_call_fn_2463011│
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
 zЄtrace_0zѕtrace_1
Џ
Ѕtrace_0
іtrace_12Я
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2463029
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2463047│
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
 zЅtrace_0zіtrace_1
 "
trackable_list_wrapper
*:(2batch_normalization_10/gamma
):'2batch_normalization_10/beta
2:0 (2"batch_normalization_10/moving_mean
6:4 (2&batch_normalization_10/moving_variance
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
8
ё0
Ћ1
є2"
trackable_list_wrapper
0
ё0
Ћ1"
trackable_list_wrapper
 "
trackable_list_wrapper
х
Іnon_trainable_variables
їlayers
Їmetrics
 јlayer_regularization_losses
Јlayer_metrics
}	variables
~trainable_variables
regularization_losses
Ђ__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses"
_generic_user_object
в
љtrace_0
Љtrace_12░
;__inference_spectral_normalization_19_layer_call_fn_2463056
;__inference_spectral_normalization_19_layer_call_fn_2463067│
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
 zљtrace_0zЉtrace_1
А
њtrace_0
Њtrace_12Т
V__inference_spectral_normalization_19_layer_call_and_return_conditional_losses_2463078
V__inference_spectral_normalization_19_layer_call_and_return_conditional_losses_2463118│
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
 zњtrace_0zЊtrace_1
Т
ћ	variables
Ћtrainable_variables
ќregularization_losses
Ќ	keras_api
ў__call__
+Ў&call_and_return_all_conditional_losses
ёkernel
	Ћbias
!џ_jit_compiled_convolution_op"
_tf_keras_layer
::82 spectral_normalization_19/kernel
 "
trackable_list_wrapper
.:,2spectral_normalization_19/sn_u
@
ј0
Ј1
љ2
Љ3"
trackable_list_wrapper
0
ј0
Ј1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Џnon_trainable_variables
юlayers
Юmetrics
 ъlayer_regularization_losses
Ъlayer_metrics
Є	variables
ѕtrainable_variables
Ѕregularization_losses
І__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
т
аtrace_0
Аtrace_12ф
8__inference_batch_normalization_11_layer_call_fn_2463131
8__inference_batch_normalization_11_layer_call_fn_2463144│
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
 zаtrace_0zАtrace_1
Џ
бtrace_0
Бtrace_12Я
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2463162
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2463180│
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
 zбtrace_0zБtrace_1
 "
trackable_list_wrapper
*:(2batch_normalization_11/gamma
):'2batch_normalization_11/beta
2:0 (2"batch_normalization_11/moving_mean
6:4 (2&batch_normalization_11/moving_variance
,:*@2spectral_normalization_16/bias
,:* 2spectral_normalization_17/bias
,:*2spectral_normalization_18/bias
,:*2spectral_normalization_19/bias
y
30
=1
>2
N3
X4
Y5
i6
s7
t8
є9
љ10
Љ11"
trackable_list_wrapper
Й
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
13
14
15
16
17
18
19
20"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
чBЭ
)__inference_model_2_layer_call_fn_2461300input_5"┐
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
)__inference_model_2_layer_call_fn_2462125inputs"┐
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
)__inference_model_2_layer_call_fn_2462186inputs"┐
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
чBЭ
)__inference_model_2_layer_call_fn_2461825input_5"┐
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
D__inference_model_2_layer_call_and_return_conditional_losses_2462324inputs"┐
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
D__inference_model_2_layer_call_and_return_conditional_losses_2462578inputs"┐
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
ќBЊ
D__inference_model_2_layer_call_and_return_conditional_losses_2461917input_5"┐
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
ќBЊ
D__inference_model_2_layer_call_and_return_conditional_losses_2462017input_5"┐
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
╠B╔
%__inference_signature_wrapper_2462072input_5"ћ
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
▀B▄
+__inference_reshape_2_layer_call_fn_2462583inputs"б
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
ЩBэ
F__inference_reshape_2_layer_call_and_return_conditional_losses_2462597inputs"б
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
тBР
1__inference_up_sampling2d_6_layer_call_fn_2462602inputs"б
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
ђB§
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_2462614inputs"б
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
30"
trackable_list_wrapper
'
00"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ђB§
;__inference_spectral_normalization_16_layer_call_fn_2462623inputs"│
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
ђB§
;__inference_spectral_normalization_16_layer_call_fn_2462634inputs"│
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
ЏBў
V__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_2462645inputs"│
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
ЏBў
V__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_2462685inputs"│
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
10
њ1"
trackable_list_wrapper
/
10
њ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
цnon_trainable_variables
Цlayers
дmetrics
 Дlayer_regularization_losses
еlayer_metrics
╗	variables
╝trainable_variables
йregularization_losses
┐__call__
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses"
_generic_user_object
е2Цб
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
е2Цб
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
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЧBщ
7__inference_batch_normalization_8_layer_call_fn_2462698inputs"│
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
ЧBщ
7__inference_batch_normalization_8_layer_call_fn_2462711inputs"│
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
ЌBћ
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2462729inputs"│
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
ЌBћ
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2462747inputs"│
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
тBР
1__inference_up_sampling2d_7_layer_call_fn_2462752inputs"б
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
ђB§
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_2462764inputs"б
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
N0"
trackable_list_wrapper
'
K0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ђB§
;__inference_spectral_normalization_17_layer_call_fn_2462773inputs"│
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
ђB§
;__inference_spectral_normalization_17_layer_call_fn_2462784inputs"│
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
ЏBў
V__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_2462795inputs"│
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
ЏBў
V__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_2462835inputs"│
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
L0
Њ1"
trackable_list_wrapper
/
L0
Њ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Еnon_trainable_variables
фlayers
Фmetrics
 гlayer_regularization_losses
Гlayer_metrics
█	variables
▄trainable_variables
Пregularization_losses
▀__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
е2Цб
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
е2Цб
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
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЧBщ
7__inference_batch_normalization_9_layer_call_fn_2462848inputs"│
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
ЧBщ
7__inference_batch_normalization_9_layer_call_fn_2462861inputs"│
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
ЌBћ
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2462879inputs"│
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
ЌBћ
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2462897inputs"│
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
тBР
1__inference_up_sampling2d_8_layer_call_fn_2462902inputs"б
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
ђB§
L__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_2462914inputs"б
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
i0"
trackable_list_wrapper
'
f0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ђB§
;__inference_spectral_normalization_18_layer_call_fn_2462923inputs"│
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
ђB§
;__inference_spectral_normalization_18_layer_call_fn_2462934inputs"│
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
ЏBў
V__inference_spectral_normalization_18_layer_call_and_return_conditional_losses_2462945inputs"│
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
ЏBў
V__inference_spectral_normalization_18_layer_call_and_return_conditional_losses_2462985inputs"│
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
g0
ћ1"
trackable_list_wrapper
/
g0
ћ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
«non_trainable_variables
»layers
░metrics
 ▒layer_regularization_losses
▓layer_metrics
ч	variables
Чtrainable_variables
§regularization_losses
 __call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
е2Цб
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
е2Цб
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
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
§BЩ
8__inference_batch_normalization_10_layer_call_fn_2462998inputs"│
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
§BЩ
8__inference_batch_normalization_10_layer_call_fn_2463011inputs"│
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
ўBЋ
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2463029inputs"│
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
ўBЋ
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2463047inputs"│
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
(
є0"
trackable_list_wrapper
(
Ѓ0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ђB§
;__inference_spectral_normalization_19_layer_call_fn_2463056inputs"│
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
ђB§
;__inference_spectral_normalization_19_layer_call_fn_2463067inputs"│
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
ЏBў
V__inference_spectral_normalization_19_layer_call_and_return_conditional_losses_2463078inputs"│
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
ЏBў
V__inference_spectral_normalization_19_layer_call_and_return_conditional_losses_2463118inputs"│
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
ё0
Ћ1"
trackable_list_wrapper
0
ё0
Ћ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
│non_trainable_variables
┤layers
хmetrics
 Хlayer_regularization_losses
иlayer_metrics
ћ	variables
Ћtrainable_variables
ќregularization_losses
ў__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
е2Цб
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
е2Цб
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
љ0
Љ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
§BЩ
8__inference_batch_normalization_11_layer_call_fn_2463131inputs"│
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
§BЩ
8__inference_batch_normalization_11_layer_call_fn_2463144inputs"│
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
ўBЋ
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2463162inputs"│
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
ўBЋ
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2463180inputs"│
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
trackable_dict_wrapperо
"__inference__wrapped_model_2460777»!1њ;<=>LЊVWXYgћqrstёЋјЈљЉ1б.
'б$
"і
input_5         ђ
ф "WфT
R
batch_normalization_118і5
batch_normalization_11         <Ь
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2463029ќqrstMбJ
Cб@
:і7
inputs+                           
p 
ф "?б<
5і2
0+                           
џ Ь
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_2463047ќqrstMбJ
Cб@
:і7
inputs+                           
p
ф "?б<
5і2
0+                           
џ к
8__inference_batch_normalization_10_layer_call_fn_2462998ЅqrstMбJ
Cб@
:і7
inputs+                           
p 
ф "2і/+                           к
8__inference_batch_normalization_10_layer_call_fn_2463011ЅqrstMбJ
Cб@
:і7
inputs+                           
p
ф "2і/+                           Ы
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2463162џјЈљЉMбJ
Cб@
:і7
inputs+                           
p 
ф "?б<
5і2
0+                           
џ Ы
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_2463180џјЈљЉMбJ
Cб@
:і7
inputs+                           
p
ф "?б<
5і2
0+                           
џ ╩
8__inference_batch_normalization_11_layer_call_fn_2463131ЇјЈљЉMбJ
Cб@
:і7
inputs+                           
p 
ф "2і/+                           ╩
8__inference_batch_normalization_11_layer_call_fn_2463144ЇјЈљЉMбJ
Cб@
:і7
inputs+                           
p
ф "2і/+                           ь
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2462729ќ;<=>MбJ
Cб@
:і7
inputs+                           @
p 
ф "?б<
5і2
0+                           @
џ ь
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2462747ќ;<=>MбJ
Cб@
:і7
inputs+                           @
p
ф "?б<
5і2
0+                           @
џ ┼
7__inference_batch_normalization_8_layer_call_fn_2462698Ѕ;<=>MбJ
Cб@
:і7
inputs+                           @
p 
ф "2і/+                           @┼
7__inference_batch_normalization_8_layer_call_fn_2462711Ѕ;<=>MбJ
Cб@
:і7
inputs+                           @
p
ф "2і/+                           @ь
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2462879ќVWXYMбJ
Cб@
:і7
inputs+                            
p 
ф "?б<
5і2
0+                            
џ ь
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_2462897ќVWXYMбJ
Cб@
:і7
inputs+                            
p
ф "?б<
5і2
0+                            
џ ┼
7__inference_batch_normalization_9_layer_call_fn_2462848ЅVWXYMбJ
Cб@
:і7
inputs+                            
p 
ф "2і/+                            ┼
7__inference_batch_normalization_9_layer_call_fn_2462861ЅVWXYMбJ
Cб@
:і7
inputs+                            
p
ф "2і/+                            У
D__inference_model_2_layer_call_and_return_conditional_losses_2461917Ъ!1њ;<=>LЊVWXYgћqrstёЋјЈљЉ9б6
/б,
"і
input_5         ђ
p 

 
ф "?б<
5і2
0+                           
џ ь
D__inference_model_2_layer_call_and_return_conditional_losses_2462017ц&13њ;<=>LNЊVWXYgiћqrstёєЋјЈљЉ9б6
/б,
"і
input_5         ђ
p

 
ф "?б<
5і2
0+                           
џ Н
D__inference_model_2_layer_call_and_return_conditional_losses_2462324ї!1њ;<=>LЊVWXYgћqrstёЋјЈљЉ8б5
.б+
!і
inputs         ђ
p 

 
ф "-б*
#і 
0         <
џ ┌
D__inference_model_2_layer_call_and_return_conditional_losses_2462578Љ&13њ;<=>LNЊVWXYgiћqrstёєЋјЈљЉ8б5
.б+
!і
inputs         ђ
p

 
ф "-б*
#і 
0         <
џ └
)__inference_model_2_layer_call_fn_2461300њ!1њ;<=>LЊVWXYgћqrstёЋјЈљЉ9б6
/б,
"і
input_5         ђ
p 

 
ф "2і/+                           ┼
)__inference_model_2_layer_call_fn_2461825Ќ&13њ;<=>LNЊVWXYgiћqrstёєЋјЈљЉ9б6
/б,
"і
input_5         ђ
p

 
ф "2і/+                           ┐
)__inference_model_2_layer_call_fn_2462125Љ!1њ;<=>LЊVWXYgћqrstёЋјЈљЉ8б5
.б+
!і
inputs         ђ
p 

 
ф "2і/+                           ─
)__inference_model_2_layer_call_fn_2462186ќ&13њ;<=>LNЊVWXYgiћqrstёєЋјЈљЉ8б5
.б+
!і
inputs         ђ
p

 
ф "2і/+                           г
F__inference_reshape_2_layer_call_and_return_conditional_losses_2462597b0б-
&б#
!і
inputs         ђ
ф ".б+
$і!
0         ђ
џ ё
+__inference_reshape_2_layer_call_fn_2462583U0б-
&б#
!і
inputs         ђ
ф "!і         ђС
%__inference_signature_wrapper_2462072║!1њ;<=>LЊVWXYgћqrstёЋјЈљЉ<б9
б 
2ф/
-
input_5"і
input_5         ђ"WфT
R
batch_normalization_118і5
batch_normalization_11         <ы
V__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_2462645ќ1њNбK
DбA
;і8
inputs,                           ђ
p 
ф "?б<
5і2
0+                           @
џ Ы
V__inference_spectral_normalization_16_layer_call_and_return_conditional_losses_2462685Ќ13њNбK
DбA
;і8
inputs,                           ђ
p
ф "?б<
5і2
0+                           @
џ ╔
;__inference_spectral_normalization_16_layer_call_fn_2462623Ѕ1њNбK
DбA
;і8
inputs,                           ђ
p 
ф "2і/+                           @╩
;__inference_spectral_normalization_16_layer_call_fn_2462634і13њNбK
DбA
;і8
inputs,                           ђ
p
ф "2і/+                           @­
V__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_2462795ЋLЊMбJ
Cб@
:і7
inputs+                           @
p 
ф "?б<
5і2
0+                            
џ ы
V__inference_spectral_normalization_17_layer_call_and_return_conditional_losses_2462835ќLNЊMбJ
Cб@
:і7
inputs+                           @
p
ф "?б<
5і2
0+                            
џ ╚
;__inference_spectral_normalization_17_layer_call_fn_2462773ѕLЊMбJ
Cб@
:і7
inputs+                           @
p 
ф "2і/+                            ╔
;__inference_spectral_normalization_17_layer_call_fn_2462784ЅLNЊMбJ
Cб@
:і7
inputs+                           @
p
ф "2і/+                            ­
V__inference_spectral_normalization_18_layer_call_and_return_conditional_losses_2462945ЋgћMбJ
Cб@
:і7
inputs+                            
p 
ф "?б<
5і2
0+                           
џ ы
V__inference_spectral_normalization_18_layer_call_and_return_conditional_losses_2462985ќgiћMбJ
Cб@
:і7
inputs+                            
p
ф "?б<
5і2
0+                           
џ ╚
;__inference_spectral_normalization_18_layer_call_fn_2462923ѕgћMбJ
Cб@
:і7
inputs+                            
p 
ф "2і/+                           ╔
;__inference_spectral_normalization_18_layer_call_fn_2462934ЅgiћMбJ
Cб@
:і7
inputs+                            
p
ф "2і/+                           ы
V__inference_spectral_normalization_19_layer_call_and_return_conditional_losses_2463078ќёЋMбJ
Cб@
:і7
inputs+                           
p 
ф "?б<
5і2
0+                           
џ з
V__inference_spectral_normalization_19_layer_call_and_return_conditional_losses_2463118ўёєЋMбJ
Cб@
:і7
inputs+                           
p
ф "?б<
5і2
0+                           
џ ╔
;__inference_spectral_normalization_19_layer_call_fn_2463056ЅёЋMбJ
Cб@
:і7
inputs+                           
p 
ф "2і/+                           ╦
;__inference_spectral_normalization_19_layer_call_fn_2463067ІёєЋMбJ
Cб@
:і7
inputs+                           
p
ф "2і/+                           №
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_2462614ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ К
1__inference_up_sampling2d_6_layer_call_fn_2462602ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    №
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_2462764ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ К
1__inference_up_sampling2d_7_layer_call_fn_2462752ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    №
L__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_2462914ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ К
1__inference_up_sampling2d_8_layer_call_fn_2462902ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    
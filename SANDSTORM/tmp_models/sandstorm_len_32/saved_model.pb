╔▄
МХ
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
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
ї
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
є
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ѕ
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.10.02unknown8па
ќ
Adam/prediction_output_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/prediction_output_1/bias/v
Ј
3Adam/prediction_output_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/prediction_output_1/bias/v*
_output_shapes
:*
dtype0
ъ
!Adam/prediction_output_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/prediction_output_1/kernel/v
Ќ
5Adam/prediction_output_1/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/prediction_output_1/kernel/v*
_output_shapes

:*
dtype0
ќ
Adam/prediction_output_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/prediction_output_0/bias/v
Ј
3Adam/prediction_output_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/prediction_output_0/bias/v*
_output_shapes
:*
dtype0
ъ
!Adam/prediction_output_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/prediction_output_0/kernel/v
Ќ
5Adam/prediction_output_0/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/prediction_output_0/kernel/v*
_output_shapes

:*
dtype0
ђ
Adam/dense_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_23/bias/v
y
(Adam/dense_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/v*
_output_shapes
:*
dtype0
ѕ
Adam/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_23/kernel/v
Ђ
*Adam/dense_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/v*
_output_shapes

:*
dtype0
ђ
Adam/dense_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_22/bias/v
y
(Adam/dense_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/v*
_output_shapes
:*
dtype0
ѕ
Adam/dense_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_22/kernel/v
Ђ
*Adam/dense_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/v*
_output_shapes

:*
dtype0
ђ
Adam/dense_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_21/bias/v
y
(Adam/dense_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/v*
_output_shapes
:*
dtype0
ѕ
Adam/dense_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:z*'
shared_nameAdam/dense_21/kernel/v
Ђ
*Adam/dense_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/v*
_output_shapes

:z*
dtype0
ѓ
Adam/conv2d_44/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_44/bias/v
{
)Adam/conv2d_44/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_44/bias/v*
_output_shapes
:*
dtype0
њ
Adam/conv2d_44/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_44/kernel/v
І
+Adam/conv2d_44/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_44/kernel/v*&
_output_shapes
:*
dtype0
ѓ
Adam/conv2d_43/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_43/bias/v
{
)Adam/conv2d_43/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_43/bias/v*
_output_shapes
:*
dtype0
њ
Adam/conv2d_43/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameAdam/conv2d_43/kernel/v
І
+Adam/conv2d_43/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_43/kernel/v*&
_output_shapes
:	*
dtype0
ѓ
Adam/conv2d_47/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_47/bias/v
{
)Adam/conv2d_47/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/bias/v*
_output_shapes
:*
dtype0
њ
Adam/conv2d_47/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_47/kernel/v
І
+Adam/conv2d_47/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/kernel/v*&
_output_shapes
:*
dtype0
џ
!Adam/batch_normalization_7/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_7/beta/v
Њ
5Adam/batch_normalization_7/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/v*
_output_shapes
:*
dtype0
ю
"Adam/batch_normalization_7/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_7/gamma/v
Ћ
6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/v*
_output_shapes
:*
dtype0
ѓ
Adam/conv2d_46/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_46/bias/v
{
)Adam/conv2d_46/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_46/bias/v*
_output_shapes
:*
dtype0
њ
Adam/conv2d_46/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_46/kernel/v
І
+Adam/conv2d_46/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_46/kernel/v*&
_output_shapes
:*
dtype0
ѓ
Adam/conv2d_42/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_42/bias/v
{
)Adam/conv2d_42/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_42/bias/v*
_output_shapes
:*
dtype0
њ
Adam/conv2d_42/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_42/kernel/v
І
+Adam/conv2d_42/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_42/kernel/v*&
_output_shapes
:*
dtype0
ѓ
Adam/conv2d_45/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_45/bias/v
{
)Adam/conv2d_45/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/bias/v*
_output_shapes
:*
dtype0
њ
Adam/conv2d_45/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*(
shared_nameAdam/conv2d_45/kernel/v
І
+Adam/conv2d_45/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/kernel/v*&
_output_shapes
:		*
dtype0
ќ
Adam/prediction_output_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/prediction_output_1/bias/m
Ј
3Adam/prediction_output_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/prediction_output_1/bias/m*
_output_shapes
:*
dtype0
ъ
!Adam/prediction_output_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/prediction_output_1/kernel/m
Ќ
5Adam/prediction_output_1/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/prediction_output_1/kernel/m*
_output_shapes

:*
dtype0
ќ
Adam/prediction_output_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/prediction_output_0/bias/m
Ј
3Adam/prediction_output_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/prediction_output_0/bias/m*
_output_shapes
:*
dtype0
ъ
!Adam/prediction_output_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/prediction_output_0/kernel/m
Ќ
5Adam/prediction_output_0/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/prediction_output_0/kernel/m*
_output_shapes

:*
dtype0
ђ
Adam/dense_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_23/bias/m
y
(Adam/dense_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/m*
_output_shapes
:*
dtype0
ѕ
Adam/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_23/kernel/m
Ђ
*Adam/dense_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/m*
_output_shapes

:*
dtype0
ђ
Adam/dense_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_22/bias/m
y
(Adam/dense_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/m*
_output_shapes
:*
dtype0
ѕ
Adam/dense_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_22/kernel/m
Ђ
*Adam/dense_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/m*
_output_shapes

:*
dtype0
ђ
Adam/dense_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_21/bias/m
y
(Adam/dense_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/m*
_output_shapes
:*
dtype0
ѕ
Adam/dense_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:z*'
shared_nameAdam/dense_21/kernel/m
Ђ
*Adam/dense_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/m*
_output_shapes

:z*
dtype0
ѓ
Adam/conv2d_44/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_44/bias/m
{
)Adam/conv2d_44/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_44/bias/m*
_output_shapes
:*
dtype0
њ
Adam/conv2d_44/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_44/kernel/m
І
+Adam/conv2d_44/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_44/kernel/m*&
_output_shapes
:*
dtype0
ѓ
Adam/conv2d_43/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_43/bias/m
{
)Adam/conv2d_43/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_43/bias/m*
_output_shapes
:*
dtype0
њ
Adam/conv2d_43/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameAdam/conv2d_43/kernel/m
І
+Adam/conv2d_43/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_43/kernel/m*&
_output_shapes
:	*
dtype0
ѓ
Adam/conv2d_47/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_47/bias/m
{
)Adam/conv2d_47/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/bias/m*
_output_shapes
:*
dtype0
њ
Adam/conv2d_47/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_47/kernel/m
І
+Adam/conv2d_47/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/kernel/m*&
_output_shapes
:*
dtype0
џ
!Adam/batch_normalization_7/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_7/beta/m
Њ
5Adam/batch_normalization_7/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/m*
_output_shapes
:*
dtype0
ю
"Adam/batch_normalization_7/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_7/gamma/m
Ћ
6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/m*
_output_shapes
:*
dtype0
ѓ
Adam/conv2d_46/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_46/bias/m
{
)Adam/conv2d_46/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_46/bias/m*
_output_shapes
:*
dtype0
њ
Adam/conv2d_46/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_46/kernel/m
І
+Adam/conv2d_46/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_46/kernel/m*&
_output_shapes
:*
dtype0
ѓ
Adam/conv2d_42/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_42/bias/m
{
)Adam/conv2d_42/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_42/bias/m*
_output_shapes
:*
dtype0
њ
Adam/conv2d_42/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_42/kernel/m
І
+Adam/conv2d_42/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_42/kernel/m*&
_output_shapes
:*
dtype0
ѓ
Adam/conv2d_45/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_45/bias/m
{
)Adam/conv2d_45/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/bias/m*
_output_shapes
:*
dtype0
њ
Adam/conv2d_45/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*(
shared_nameAdam/conv2d_45/kernel/m
І
+Adam/conv2d_45/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/kernel/m*&
_output_shapes
:		*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
ѕ
prediction_output_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameprediction_output_1/bias
Ђ
,prediction_output_1/bias/Read/ReadVariableOpReadVariableOpprediction_output_1/bias*
_output_shapes
:*
dtype0
љ
prediction_output_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameprediction_output_1/kernel
Ѕ
.prediction_output_1/kernel/Read/ReadVariableOpReadVariableOpprediction_output_1/kernel*
_output_shapes

:*
dtype0
ѕ
prediction_output_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameprediction_output_0/bias
Ђ
,prediction_output_0/bias/Read/ReadVariableOpReadVariableOpprediction_output_0/bias*
_output_shapes
:*
dtype0
љ
prediction_output_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameprediction_output_0/kernel
Ѕ
.prediction_output_0/kernel/Read/ReadVariableOpReadVariableOpprediction_output_0/kernel*
_output_shapes

:*
dtype0
r
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes
:*
dtype0
z
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_23/kernel
s
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes

:*
dtype0
r
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_22/bias
k
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes
:*
dtype0
z
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_22/kernel
s
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel*
_output_shapes

:*
dtype0
r
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_21/bias
k
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes
:*
dtype0
z
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:z* 
shared_namedense_21/kernel
s
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes

:z*
dtype0
t
conv2d_44/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_44/bias
m
"conv2d_44/bias/Read/ReadVariableOpReadVariableOpconv2d_44/bias*
_output_shapes
:*
dtype0
ё
conv2d_44/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_44/kernel
}
$conv2d_44/kernel/Read/ReadVariableOpReadVariableOpconv2d_44/kernel*&
_output_shapes
:*
dtype0
t
conv2d_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_43/bias
m
"conv2d_43/bias/Read/ReadVariableOpReadVariableOpconv2d_43/bias*
_output_shapes
:*
dtype0
ё
conv2d_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*!
shared_nameconv2d_43/kernel
}
$conv2d_43/kernel/Read/ReadVariableOpReadVariableOpconv2d_43/kernel*&
_output_shapes
:	*
dtype0
t
conv2d_47/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_47/bias
m
"conv2d_47/bias/Read/ReadVariableOpReadVariableOpconv2d_47/bias*
_output_shapes
:*
dtype0
ё
conv2d_47/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_47/kernel
}
$conv2d_47/kernel/Read/ReadVariableOpReadVariableOpconv2d_47/kernel*&
_output_shapes
:*
dtype0
б
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_7/moving_variance
Џ
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
:*
dtype0
џ
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_7/moving_mean
Њ
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
:*
dtype0
ї
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_7/beta
Ё
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
:*
dtype0
ј
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_7/gamma
Є
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
:*
dtype0
t
conv2d_46/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_46/bias
m
"conv2d_46/bias/Read/ReadVariableOpReadVariableOpconv2d_46/bias*
_output_shapes
:*
dtype0
ё
conv2d_46/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_46/kernel
}
$conv2d_46/kernel/Read/ReadVariableOpReadVariableOpconv2d_46/kernel*&
_output_shapes
:*
dtype0
t
conv2d_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_42/bias
m
"conv2d_42/bias/Read/ReadVariableOpReadVariableOpconv2d_42/bias*
_output_shapes
:*
dtype0
ё
conv2d_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_42/kernel
}
$conv2d_42/kernel/Read/ReadVariableOpReadVariableOpconv2d_42/kernel*&
_output_shapes
:*
dtype0
t
conv2d_45/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_45/bias
m
"conv2d_45/bias/Read/ReadVariableOpReadVariableOpconv2d_45/bias*
_output_shapes
:*
dtype0
ё
conv2d_45/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*!
shared_nameconv2d_45/kernel
}
$conv2d_45/kernel/Read/ReadVariableOpReadVariableOpconv2d_45/kernel*&
_output_shapes
:		*
dtype0
І
serving_default_input_15Placeholder*/
_output_shapes
:         <*
dtype0*$
shape:         <
І
serving_default_input_16Placeholder*/
_output_shapes
:         <<*
dtype0*$
shape:         <<
┤
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_15serving_default_input_16conv2d_45/kernelconv2d_45/biasconv2d_42/kernelconv2d_42/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_46/kernelconv2d_46/biasconv2d_43/kernelconv2d_43/biasconv2d_47/kernelconv2d_47/biasconv2d_44/kernelconv2d_44/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/biasprediction_output_1/kernelprediction_output_1/biasprediction_output_0/kernelprediction_output_0/bias*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         :         *<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *-
f(R&
$__inference_signature_wrapper_717349

NoOpNoOp
О░
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Љ░
valueє░Bѓ░ BЩ»
є
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer-11
layer-12
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
layer_with_weights-10
layer-17
layer_with_weights-11
layer-18
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
╚
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias
 %_jit_compiled_convolution_op*
* 
Ц
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,_random_generator* 
╚
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias
 5_jit_compiled_convolution_op*
╚
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

<kernel
=bias
 >_jit_compiled_convolution_op*
Н
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses
Eaxis
	Fgamma
Gbeta
Hmoving_mean
Imoving_variance*
╚
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses

Pkernel
Qbias
 R_jit_compiled_convolution_op*
╚
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias
 [_jit_compiled_convolution_op*
ј
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses* 
╚
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias
 j_jit_compiled_convolution_op*
ј
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses* 
ј
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses* 
ј
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses* 
Ф
}	variables
~trainable_variables
regularization_losses
ђ	keras_api
Ђ__call__
+ѓ&call_and_return_all_conditional_losses
Ѓkernel
	ёbias*
«
Ё	variables
єtrainable_variables
Єregularization_losses
ѕ	keras_api
Ѕ__call__
+і&call_and_return_all_conditional_losses
Іkernel
	їbias*
«
Ї	variables
јtrainable_variables
Јregularization_losses
љ	keras_api
Љ__call__
+њ&call_and_return_all_conditional_losses
Њkernel
	ћbias*
«
Ћ	variables
ќtrainable_variables
Ќregularization_losses
ў	keras_api
Ў__call__
+џ&call_and_return_all_conditional_losses
Џkernel
	юbias*
«
Ю	variables
ъtrainable_variables
Ъregularization_losses
а	keras_api
А__call__
+б&call_and_return_all_conditional_losses
Бkernel
	цbias*
н
#0
$1
32
43
<4
=5
F6
G7
H8
I9
P10
Q11
Y12
Z13
h14
i15
Ѓ16
ё17
І18
ї19
Њ20
ћ21
Џ22
ю23
Б24
ц25*
─
#0
$1
32
43
<4
=5
F6
G7
P8
Q9
Y10
Z11
h12
i13
Ѓ14
ё15
І16
ї17
Њ18
ћ19
Џ20
ю21
Б22
ц23*
* 
х
Цnon_trainable_variables
дlayers
Дmetrics
 еlayer_regularization_losses
Еlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
фtrace_0
Фtrace_1
гtrace_2
Гtrace_3* 
:
«trace_0
»trace_1
░trace_2
▒trace_3* 
* 
й
	▓iter
│beta_1
┤beta_2

хdecay
Хlearning_rate#m┬$m├3m─4m┼<mк=mКFm╚Gm╔Pm╩Qm╦Ym╠Zm═hm╬im¤	Ѓmл	ёmЛ	Іmм	їmМ	Њmн	ћmН	Џmо	юmО	Бmп	цm┘#v┌$v█3v▄4vП<vя=v▀FvЯGvрPvРQvсYvСZvтhvТivу	ЃvУ	ёvж	ІvЖ	їvв	ЊvВ	ћvь	ЏvЬ	юv№	Бv­	цvы*

иserving_default* 

#0
$1*

#0
$1*
* 
ў
Иnon_trainable_variables
╣layers
║metrics
 ╗layer_regularization_losses
╝layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

йtrace_0* 

Йtrace_0* 
`Z
VARIABLE_VALUEconv2d_45/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_45/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ќ
┐non_trainable_variables
└layers
┴metrics
 ┬layer_regularization_losses
├layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses* 

─trace_0
┼trace_1* 

кtrace_0
Кtrace_1* 
* 

30
41*

30
41*
* 
ў
╚non_trainable_variables
╔layers
╩metrics
 ╦layer_regularization_losses
╠layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

═trace_0* 

╬trace_0* 
`Z
VARIABLE_VALUEconv2d_42/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_42/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

<0
=1*

<0
=1*
* 
ў
¤non_trainable_variables
лlayers
Лmetrics
 мlayer_regularization_losses
Мlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*

нtrace_0* 

Нtrace_0* 
`Z
VARIABLE_VALUEconv2d_46/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_46/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
F0
G1
H2
I3*

F0
G1*
* 
ў
оnon_trainable_variables
Оlayers
пmetrics
 ┘layer_regularization_losses
┌layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*

█trace_0
▄trace_1* 

Пtrace_0
яtrace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_7/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_7/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_7/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_7/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

P0
Q1*

P0
Q1*
* 
ў
▀non_trainable_variables
Яlayers
рmetrics
 Рlayer_regularization_losses
сlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*

Сtrace_0* 

тtrace_0* 
`Z
VARIABLE_VALUEconv2d_47/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_47/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

Y0
Z1*

Y0
Z1*
* 
ў
Тnon_trainable_variables
уlayers
Уmetrics
 жlayer_regularization_losses
Жlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

вtrace_0* 

Вtrace_0* 
`Z
VARIABLE_VALUEconv2d_43/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_43/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ќ
ьnon_trainable_variables
Ьlayers
№metrics
 ­layer_regularization_losses
ыlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses* 

Ыtrace_0* 

зtrace_0* 

h0
i1*

h0
i1*
* 
ў
Зnon_trainable_variables
шlayers
Шmetrics
 эlayer_regularization_losses
Эlayer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

щtrace_0* 

Щtrace_0* 
`Z
VARIABLE_VALUEconv2d_44/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_44/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ќ
чnon_trainable_variables
Чlayers
§metrics
 ■layer_regularization_losses
 layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses* 

ђtrace_0* 

Ђtrace_0* 
* 
* 
* 
ќ
ѓnon_trainable_variables
Ѓlayers
ёmetrics
 Ёlayer_regularization_losses
єlayer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses* 

Єtrace_0* 

ѕtrace_0* 
* 
* 
* 
ќ
Ѕnon_trainable_variables
іlayers
Іmetrics
 їlayer_regularization_losses
Їlayer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses* 

јtrace_0* 

Јtrace_0* 

Ѓ0
ё1*

Ѓ0
ё1*
* 
Џ
љnon_trainable_variables
Љlayers
њmetrics
 Њlayer_regularization_losses
ћlayer_metrics
}	variables
~trainable_variables
regularization_losses
Ђ__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses*

Ћtrace_0* 

ќtrace_0* 
_Y
VARIABLE_VALUEdense_21/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_21/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

І0
ї1*

І0
ї1*
* 
ъ
Ќnon_trainable_variables
ўlayers
Ўmetrics
 џlayer_regularization_losses
Џlayer_metrics
Ё	variables
єtrainable_variables
Єregularization_losses
Ѕ__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses*

юtrace_0* 

Юtrace_0* 
_Y
VARIABLE_VALUEdense_22/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_22/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

Њ0
ћ1*

Њ0
ћ1*
* 
ъ
ъnon_trainable_variables
Ъlayers
аmetrics
 Аlayer_regularization_losses
бlayer_metrics
Ї	variables
јtrainable_variables
Јregularization_losses
Љ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses*

Бtrace_0* 

цtrace_0* 
_Y
VARIABLE_VALUEdense_23/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_23/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

Џ0
ю1*

Џ0
ю1*
* 
ъ
Цnon_trainable_variables
дlayers
Дmetrics
 еlayer_regularization_losses
Еlayer_metrics
Ћ	variables
ќtrainable_variables
Ќregularization_losses
Ў__call__
+џ&call_and_return_all_conditional_losses
'џ"call_and_return_conditional_losses*

фtrace_0* 

Фtrace_0* 
ke
VARIABLE_VALUEprediction_output_0/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEprediction_output_0/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*

Б0
ц1*

Б0
ц1*
* 
ъ
гnon_trainable_variables
Гlayers
«metrics
 »layer_regularization_losses
░layer_metrics
Ю	variables
ъtrainable_variables
Ъregularization_losses
А__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses*

▒trace_0* 

▓trace_0* 
ke
VARIABLE_VALUEprediction_output_1/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEprediction_output_1/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*

H0
I1*
њ
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
18*

│0
┤1
х2*
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
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
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

H0
I1*
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
<
Х	variables
и	keras_api

Иtotal

╣count*
<
║	variables
╗	keras_api

╝total

йcount*
<
Й	variables
┐	keras_api

└total

┴count*

И0
╣1*

Х	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

╝0
й1*

║	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

└0
┴1*

Й	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
Ѓ}
VARIABLE_VALUEAdam/conv2d_45/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_45/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Ѓ}
VARIABLE_VALUEAdam/conv2d_42/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_42/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Ѓ}
VARIABLE_VALUEAdam/conv2d_46/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_46/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
јЄ
VARIABLE_VALUE"Adam/batch_normalization_7/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
їЁ
VARIABLE_VALUE!Adam/batch_normalization_7/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Ѓ}
VARIABLE_VALUEAdam/conv2d_47/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_47/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Ѓ}
VARIABLE_VALUEAdam/conv2d_43/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_43/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Ѓ}
VARIABLE_VALUEAdam/conv2d_44/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_44/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ѓ|
VARIABLE_VALUEAdam/dense_21/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_21/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ѓ|
VARIABLE_VALUEAdam/dense_22/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_22/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ѓ|
VARIABLE_VALUEAdam/dense_23/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_23/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Јѕ
VARIABLE_VALUE!Adam/prediction_output_0/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Іё
VARIABLE_VALUEAdam/prediction_output_0/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Јѕ
VARIABLE_VALUE!Adam/prediction_output_1/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Іё
VARIABLE_VALUEAdam/prediction_output_1/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Ѓ}
VARIABLE_VALUEAdam/conv2d_45/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_45/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Ѓ}
VARIABLE_VALUEAdam/conv2d_42/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_42/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Ѓ}
VARIABLE_VALUEAdam/conv2d_46/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_46/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
јЄ
VARIABLE_VALUE"Adam/batch_normalization_7/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
їЁ
VARIABLE_VALUE!Adam/batch_normalization_7/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Ѓ}
VARIABLE_VALUEAdam/conv2d_47/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_47/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Ѓ}
VARIABLE_VALUEAdam/conv2d_43/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_43/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Ѓ}
VARIABLE_VALUEAdam/conv2d_44/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_44/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ѓ|
VARIABLE_VALUEAdam/dense_21/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_21/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ѓ|
VARIABLE_VALUEAdam/dense_22/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_22/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ѓ|
VARIABLE_VALUEAdam/dense_23/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_23/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Јѕ
VARIABLE_VALUE!Adam/prediction_output_0/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Іё
VARIABLE_VALUEAdam/prediction_output_0/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Јѕ
VARIABLE_VALUE!Adam/prediction_output_1/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Іё
VARIABLE_VALUEAdam/prediction_output_1/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Џ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_45/kernel/Read/ReadVariableOp"conv2d_45/bias/Read/ReadVariableOp$conv2d_42/kernel/Read/ReadVariableOp"conv2d_42/bias/Read/ReadVariableOp$conv2d_46/kernel/Read/ReadVariableOp"conv2d_46/bias/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp$conv2d_47/kernel/Read/ReadVariableOp"conv2d_47/bias/Read/ReadVariableOp$conv2d_43/kernel/Read/ReadVariableOp"conv2d_43/bias/Read/ReadVariableOp$conv2d_44/kernel/Read/ReadVariableOp"conv2d_44/bias/Read/ReadVariableOp#dense_21/kernel/Read/ReadVariableOp!dense_21/bias/Read/ReadVariableOp#dense_22/kernel/Read/ReadVariableOp!dense_22/bias/Read/ReadVariableOp#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOp.prediction_output_0/kernel/Read/ReadVariableOp,prediction_output_0/bias/Read/ReadVariableOp.prediction_output_1/kernel/Read/ReadVariableOp,prediction_output_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv2d_45/kernel/m/Read/ReadVariableOp)Adam/conv2d_45/bias/m/Read/ReadVariableOp+Adam/conv2d_42/kernel/m/Read/ReadVariableOp)Adam/conv2d_42/bias/m/Read/ReadVariableOp+Adam/conv2d_46/kernel/m/Read/ReadVariableOp)Adam/conv2d_46/bias/m/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_7/beta/m/Read/ReadVariableOp+Adam/conv2d_47/kernel/m/Read/ReadVariableOp)Adam/conv2d_47/bias/m/Read/ReadVariableOp+Adam/conv2d_43/kernel/m/Read/ReadVariableOp)Adam/conv2d_43/bias/m/Read/ReadVariableOp+Adam/conv2d_44/kernel/m/Read/ReadVariableOp)Adam/conv2d_44/bias/m/Read/ReadVariableOp*Adam/dense_21/kernel/m/Read/ReadVariableOp(Adam/dense_21/bias/m/Read/ReadVariableOp*Adam/dense_22/kernel/m/Read/ReadVariableOp(Adam/dense_22/bias/m/Read/ReadVariableOp*Adam/dense_23/kernel/m/Read/ReadVariableOp(Adam/dense_23/bias/m/Read/ReadVariableOp5Adam/prediction_output_0/kernel/m/Read/ReadVariableOp3Adam/prediction_output_0/bias/m/Read/ReadVariableOp5Adam/prediction_output_1/kernel/m/Read/ReadVariableOp3Adam/prediction_output_1/bias/m/Read/ReadVariableOp+Adam/conv2d_45/kernel/v/Read/ReadVariableOp)Adam/conv2d_45/bias/v/Read/ReadVariableOp+Adam/conv2d_42/kernel/v/Read/ReadVariableOp)Adam/conv2d_42/bias/v/Read/ReadVariableOp+Adam/conv2d_46/kernel/v/Read/ReadVariableOp)Adam/conv2d_46/bias/v/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_7/beta/v/Read/ReadVariableOp+Adam/conv2d_47/kernel/v/Read/ReadVariableOp)Adam/conv2d_47/bias/v/Read/ReadVariableOp+Adam/conv2d_43/kernel/v/Read/ReadVariableOp)Adam/conv2d_43/bias/v/Read/ReadVariableOp+Adam/conv2d_44/kernel/v/Read/ReadVariableOp)Adam/conv2d_44/bias/v/Read/ReadVariableOp*Adam/dense_21/kernel/v/Read/ReadVariableOp(Adam/dense_21/bias/v/Read/ReadVariableOp*Adam/dense_22/kernel/v/Read/ReadVariableOp(Adam/dense_22/bias/v/Read/ReadVariableOp*Adam/dense_23/kernel/v/Read/ReadVariableOp(Adam/dense_23/bias/v/Read/ReadVariableOp5Adam/prediction_output_0/kernel/v/Read/ReadVariableOp3Adam/prediction_output_0/bias/v/Read/ReadVariableOp5Adam/prediction_output_1/kernel/v/Read/ReadVariableOp3Adam/prediction_output_1/bias/v/Read/ReadVariableOpConst*b
Tin[
Y2W	*
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
__inference__traced_save_718339
Ы
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_45/kernelconv2d_45/biasconv2d_42/kernelconv2d_42/biasconv2d_46/kernelconv2d_46/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_47/kernelconv2d_47/biasconv2d_43/kernelconv2d_43/biasconv2d_44/kernelconv2d_44/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/biasprediction_output_0/kernelprediction_output_0/biasprediction_output_1/kernelprediction_output_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_2count_2total_1count_1totalcountAdam/conv2d_45/kernel/mAdam/conv2d_45/bias/mAdam/conv2d_42/kernel/mAdam/conv2d_42/bias/mAdam/conv2d_46/kernel/mAdam/conv2d_46/bias/m"Adam/batch_normalization_7/gamma/m!Adam/batch_normalization_7/beta/mAdam/conv2d_47/kernel/mAdam/conv2d_47/bias/mAdam/conv2d_43/kernel/mAdam/conv2d_43/bias/mAdam/conv2d_44/kernel/mAdam/conv2d_44/bias/mAdam/dense_21/kernel/mAdam/dense_21/bias/mAdam/dense_22/kernel/mAdam/dense_22/bias/mAdam/dense_23/kernel/mAdam/dense_23/bias/m!Adam/prediction_output_0/kernel/mAdam/prediction_output_0/bias/m!Adam/prediction_output_1/kernel/mAdam/prediction_output_1/bias/mAdam/conv2d_45/kernel/vAdam/conv2d_45/bias/vAdam/conv2d_42/kernel/vAdam/conv2d_42/bias/vAdam/conv2d_46/kernel/vAdam/conv2d_46/bias/v"Adam/batch_normalization_7/gamma/v!Adam/batch_normalization_7/beta/vAdam/conv2d_47/kernel/vAdam/conv2d_47/bias/vAdam/conv2d_43/kernel/vAdam/conv2d_43/bias/vAdam/conv2d_44/kernel/vAdam/conv2d_44/bias/vAdam/dense_21/kernel/vAdam/dense_21/bias/vAdam/dense_22/kernel/vAdam/dense_22/bias/vAdam/dense_23/kernel/vAdam/dense_23/bias/v!Adam/prediction_output_0/kernel/vAdam/prediction_output_0/bias/v!Adam/prediction_output_1/kernel/vAdam/prediction_output_1/bias/v*a
TinZ
X2V*
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
"__inference__traced_restore_718604ає
ЙЂ
н
G__inference_joint_model_layer_call_and_return_conditional_losses_717573
inputs_0
inputs_1B
(conv2d_45_conv2d_readvariableop_resource:		7
)conv2d_45_biasadd_readvariableop_resource:B
(conv2d_42_conv2d_readvariableop_resource:7
)conv2d_42_biasadd_readvariableop_resource:;
-batch_normalization_7_readvariableop_resource:=
/batch_normalization_7_readvariableop_1_resource:L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_46_conv2d_readvariableop_resource:7
)conv2d_46_biasadd_readvariableop_resource:B
(conv2d_43_conv2d_readvariableop_resource:	7
)conv2d_43_biasadd_readvariableop_resource:B
(conv2d_47_conv2d_readvariableop_resource:7
)conv2d_47_biasadd_readvariableop_resource:B
(conv2d_44_conv2d_readvariableop_resource:7
)conv2d_44_biasadd_readvariableop_resource:9
'dense_21_matmul_readvariableop_resource:z6
(dense_21_biasadd_readvariableop_resource:9
'dense_22_matmul_readvariableop_resource:6
(dense_22_biasadd_readvariableop_resource:9
'dense_23_matmul_readvariableop_resource:6
(dense_23_biasadd_readvariableop_resource:D
2prediction_output_1_matmul_readvariableop_resource:A
3prediction_output_1_biasadd_readvariableop_resource:D
2prediction_output_0_matmul_readvariableop_resource:A
3prediction_output_0_biasadd_readvariableop_resource:
identity

identity_1ѕб5batch_normalization_7/FusedBatchNormV3/ReadVariableOpб7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_7/ReadVariableOpб&batch_normalization_7/ReadVariableOp_1б conv2d_42/BiasAdd/ReadVariableOpбconv2d_42/Conv2D/ReadVariableOpб conv2d_43/BiasAdd/ReadVariableOpбconv2d_43/Conv2D/ReadVariableOpб conv2d_44/BiasAdd/ReadVariableOpбconv2d_44/Conv2D/ReadVariableOpб conv2d_45/BiasAdd/ReadVariableOpбconv2d_45/Conv2D/ReadVariableOpб conv2d_46/BiasAdd/ReadVariableOpбconv2d_46/Conv2D/ReadVariableOpб conv2d_47/BiasAdd/ReadVariableOpбconv2d_47/Conv2D/ReadVariableOpбdense_21/BiasAdd/ReadVariableOpбdense_21/MatMul/ReadVariableOpбdense_22/BiasAdd/ReadVariableOpбdense_22/MatMul/ReadVariableOpбdense_23/BiasAdd/ReadVariableOpбdense_23/MatMul/ReadVariableOpб*prediction_output_0/BiasAdd/ReadVariableOpб)prediction_output_0/MatMul/ReadVariableOpб*prediction_output_1/BiasAdd/ReadVariableOpб)prediction_output_1/MatMul/ReadVariableOpљ
conv2d_45/Conv2D/ReadVariableOpReadVariableOp(conv2d_45_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0»
conv2d_45/Conv2DConv2Dinputs_1'conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
є
 conv2d_45/BiasAdd/ReadVariableOpReadVariableOp)conv2d_45_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Џ
conv2d_45/BiasAddBiasAddconv2d_45/Conv2D:output:0(conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         l
conv2d_45/ReluReluconv2d_45/BiasAdd:output:0*
T0*/
_output_shapes
:         љ
conv2d_42/Conv2D/ReadVariableOpReadVariableOp(conv2d_42_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0»
conv2d_42/Conv2DConv2Dinputs_0'conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <*
paddingSAME*
strides
є
 conv2d_42/BiasAdd/ReadVariableOpReadVariableOp)conv2d_42_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Џ
conv2d_42/BiasAddBiasAddconv2d_42/Conv2D:output:0(conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <l
conv2d_42/ReluReluconv2d_42/BiasAdd:output:0*
T0*/
_output_shapes
:         <ђ
spatial_dropout2d_7/IdentityIdentityconv2d_45/Relu:activations:0*
T0*/
_output_shapes
:         ј
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype0њ
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype0░
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0┤
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0║
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_42/Relu:activations:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         <:::::*
epsilon%oЃ:*
is_training( љ
conv2d_46/Conv2D/ReadVariableOpReadVariableOp(conv2d_46_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╠
conv2d_46/Conv2DConv2D%spatial_dropout2d_7/Identity:output:0'conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
є
 conv2d_46/BiasAdd/ReadVariableOpReadVariableOp)conv2d_46_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Џ
conv2d_46/BiasAddBiasAddconv2d_46/Conv2D:output:0(conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         l
conv2d_46/ReluReluconv2d_46/BiasAdd:output:0*
T0*/
_output_shapes
:         љ
conv2d_43/Conv2D/ReadVariableOpReadVariableOp(conv2d_43_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0Л
conv2d_43/Conv2DConv2D*batch_normalization_7/FusedBatchNormV3:y:0'conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <*
paddingSAME*
strides
є
 conv2d_43/BiasAdd/ReadVariableOpReadVariableOp)conv2d_43_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Џ
conv2d_43/BiasAddBiasAddconv2d_43/Conv2D:output:0(conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <l
conv2d_43/ReluReluconv2d_43/BiasAdd:output:0*
T0*/
_output_shapes
:         <љ
conv2d_47/Conv2D/ReadVariableOpReadVariableOp(conv2d_47_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0├
conv2d_47/Conv2DConv2Dconv2d_46/Relu:activations:0'conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
є
 conv2d_47/BiasAdd/ReadVariableOpReadVariableOp)conv2d_47_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Џ
conv2d_47/BiasAddBiasAddconv2d_47/Conv2D:output:0(conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         l
conv2d_47/ReluReluconv2d_47/BiasAdd:output:0*
T0*/
_output_shapes
:         љ
conv2d_44/Conv2D/ReadVariableOpReadVariableOp(conv2d_44_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0├
conv2d_44/Conv2DConv2Dconv2d_43/Relu:activations:0'conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <*
paddingSAME*
strides
є
 conv2d_44/BiasAdd/ReadVariableOpReadVariableOp)conv2d_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Џ
conv2d_44/BiasAddBiasAddconv2d_44/Conv2D:output:0(conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <l
conv2d_44/ReluReluconv2d_44/BiasAdd:output:0*
T0*/
_output_shapes
:         <}
,global_max_pooling2d_7/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      е
global_max_pooling2d_7/MaxMaxconv2d_47/Relu:activations:05global_max_pooling2d_7/Max/reduction_indices:output:0*
T0*'
_output_shapes
:         a
flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ј
flatten_15/ReshapeReshape#global_max_pooling2d_7/Max:output:0flatten_15/Const:output:0*
T0*'
_output_shapes
:         a
flatten_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"    x   ѕ
flatten_14/ReshapeReshapeconv2d_44/Relu:activations:0flatten_14/Const:output:0*
T0*'
_output_shapes
:         x[
concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :╣
concatenate_7/concatConcatV2flatten_15/Reshape:output:0flatten_14/Reshape:output:0"concatenate_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:         zє
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

:z*
dtype0њ
dense_21/MatMulMatMulconcatenate_7/concat:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         b
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*'
_output_shapes
:         є
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:*
dtype0љ
dense_22/MatMulMatMuldense_21/Relu:activations:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         b
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*'
_output_shapes
:         є
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:*
dtype0љ
dense_23/MatMulMatMuldense_22/Relu:activations:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         b
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*'
_output_shapes
:         ю
)prediction_output_1/MatMul/ReadVariableOpReadVariableOp2prediction_output_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0д
prediction_output_1/MatMulMatMuldense_23/Relu:activations:01prediction_output_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         џ
*prediction_output_1/BiasAdd/ReadVariableOpReadVariableOp3prediction_output_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0▓
prediction_output_1/BiasAddBiasAdd$prediction_output_1/MatMul:product:02prediction_output_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ю
)prediction_output_0/MatMul/ReadVariableOpReadVariableOp2prediction_output_0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0д
prediction_output_0/MatMulMatMuldense_23/Relu:activations:01prediction_output_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         џ
*prediction_output_0/BiasAdd/ReadVariableOpReadVariableOp3prediction_output_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0▓
prediction_output_0/BiasAddBiasAdd$prediction_output_0/MatMul:product:02prediction_output_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         s
IdentityIdentity$prediction_output_0/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         u

Identity_1Identity$prediction_output_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         А
NoOpNoOp6^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1!^conv2d_42/BiasAdd/ReadVariableOp ^conv2d_42/Conv2D/ReadVariableOp!^conv2d_43/BiasAdd/ReadVariableOp ^conv2d_43/Conv2D/ReadVariableOp!^conv2d_44/BiasAdd/ReadVariableOp ^conv2d_44/Conv2D/ReadVariableOp!^conv2d_45/BiasAdd/ReadVariableOp ^conv2d_45/Conv2D/ReadVariableOp!^conv2d_46/BiasAdd/ReadVariableOp ^conv2d_46/Conv2D/ReadVariableOp!^conv2d_47/BiasAdd/ReadVariableOp ^conv2d_47/Conv2D/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp+^prediction_output_0/BiasAdd/ReadVariableOp*^prediction_output_0/MatMul/ReadVariableOp+^prediction_output_1/BiasAdd/ReadVariableOp*^prediction_output_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:         <:         <<: : : : : : : : : : : : : : : : : : : : : : : : : : 2n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12D
 conv2d_42/BiasAdd/ReadVariableOp conv2d_42/BiasAdd/ReadVariableOp2B
conv2d_42/Conv2D/ReadVariableOpconv2d_42/Conv2D/ReadVariableOp2D
 conv2d_43/BiasAdd/ReadVariableOp conv2d_43/BiasAdd/ReadVariableOp2B
conv2d_43/Conv2D/ReadVariableOpconv2d_43/Conv2D/ReadVariableOp2D
 conv2d_44/BiasAdd/ReadVariableOp conv2d_44/BiasAdd/ReadVariableOp2B
conv2d_44/Conv2D/ReadVariableOpconv2d_44/Conv2D/ReadVariableOp2D
 conv2d_45/BiasAdd/ReadVariableOp conv2d_45/BiasAdd/ReadVariableOp2B
conv2d_45/Conv2D/ReadVariableOpconv2d_45/Conv2D/ReadVariableOp2D
 conv2d_46/BiasAdd/ReadVariableOp conv2d_46/BiasAdd/ReadVariableOp2B
conv2d_46/Conv2D/ReadVariableOpconv2d_46/Conv2D/ReadVariableOp2D
 conv2d_47/BiasAdd/ReadVariableOp conv2d_47/BiasAdd/ReadVariableOp2B
conv2d_47/Conv2D/ReadVariableOpconv2d_47/Conv2D/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2X
*prediction_output_0/BiasAdd/ReadVariableOp*prediction_output_0/BiasAdd/ReadVariableOp2V
)prediction_output_0/MatMul/ReadVariableOp)prediction_output_0/MatMul/ReadVariableOp2X
*prediction_output_1/BiasAdd/ReadVariableOp*prediction_output_1/BiasAdd/ReadVariableOp2V
)prediction_output_1/MatMul/ReadVariableOp)prediction_output_1/MatMul/ReadVariableOp:Y U
/
_output_shapes
:         <
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         <<
"
_user_specified_name
inputs/1
ё
■
E__inference_conv2d_46_layer_call_and_return_conditional_losses_717793

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
№
Ъ
*__inference_conv2d_43_layer_call_fn_717884

inputs!
unknown:	
	unknown_0:
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         <*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_43_layer_call_and_return_conditional_losses_716539w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         <`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         <: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         <
 
_user_specified_nameinputs
░а
ц
G__inference_joint_model_layer_call_and_return_conditional_losses_717695
inputs_0
inputs_1B
(conv2d_45_conv2d_readvariableop_resource:		7
)conv2d_45_biasadd_readvariableop_resource:B
(conv2d_42_conv2d_readvariableop_resource:7
)conv2d_42_biasadd_readvariableop_resource:;
-batch_normalization_7_readvariableop_resource:=
/batch_normalization_7_readvariableop_1_resource:L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_46_conv2d_readvariableop_resource:7
)conv2d_46_biasadd_readvariableop_resource:B
(conv2d_43_conv2d_readvariableop_resource:	7
)conv2d_43_biasadd_readvariableop_resource:B
(conv2d_47_conv2d_readvariableop_resource:7
)conv2d_47_biasadd_readvariableop_resource:B
(conv2d_44_conv2d_readvariableop_resource:7
)conv2d_44_biasadd_readvariableop_resource:9
'dense_21_matmul_readvariableop_resource:z6
(dense_21_biasadd_readvariableop_resource:9
'dense_22_matmul_readvariableop_resource:6
(dense_22_biasadd_readvariableop_resource:9
'dense_23_matmul_readvariableop_resource:6
(dense_23_biasadd_readvariableop_resource:D
2prediction_output_1_matmul_readvariableop_resource:A
3prediction_output_1_biasadd_readvariableop_resource:D
2prediction_output_0_matmul_readvariableop_resource:A
3prediction_output_0_biasadd_readvariableop_resource:
identity

identity_1ѕб$batch_normalization_7/AssignNewValueб&batch_normalization_7/AssignNewValue_1б5batch_normalization_7/FusedBatchNormV3/ReadVariableOpб7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_7/ReadVariableOpб&batch_normalization_7/ReadVariableOp_1б conv2d_42/BiasAdd/ReadVariableOpбconv2d_42/Conv2D/ReadVariableOpб conv2d_43/BiasAdd/ReadVariableOpбconv2d_43/Conv2D/ReadVariableOpб conv2d_44/BiasAdd/ReadVariableOpбconv2d_44/Conv2D/ReadVariableOpб conv2d_45/BiasAdd/ReadVariableOpбconv2d_45/Conv2D/ReadVariableOpб conv2d_46/BiasAdd/ReadVariableOpбconv2d_46/Conv2D/ReadVariableOpб conv2d_47/BiasAdd/ReadVariableOpбconv2d_47/Conv2D/ReadVariableOpбdense_21/BiasAdd/ReadVariableOpбdense_21/MatMul/ReadVariableOpбdense_22/BiasAdd/ReadVariableOpбdense_22/MatMul/ReadVariableOpбdense_23/BiasAdd/ReadVariableOpбdense_23/MatMul/ReadVariableOpб*prediction_output_0/BiasAdd/ReadVariableOpб)prediction_output_0/MatMul/ReadVariableOpб*prediction_output_1/BiasAdd/ReadVariableOpб)prediction_output_1/MatMul/ReadVariableOpљ
conv2d_45/Conv2D/ReadVariableOpReadVariableOp(conv2d_45_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0»
conv2d_45/Conv2DConv2Dinputs_1'conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
є
 conv2d_45/BiasAdd/ReadVariableOpReadVariableOp)conv2d_45_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Џ
conv2d_45/BiasAddBiasAddconv2d_45/Conv2D:output:0(conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         l
conv2d_45/ReluReluconv2d_45/BiasAdd:output:0*
T0*/
_output_shapes
:         љ
conv2d_42/Conv2D/ReadVariableOpReadVariableOp(conv2d_42_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0»
conv2d_42/Conv2DConv2Dinputs_0'conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <*
paddingSAME*
strides
є
 conv2d_42/BiasAdd/ReadVariableOpReadVariableOp)conv2d_42_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Џ
conv2d_42/BiasAddBiasAddconv2d_42/Conv2D:output:0(conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <l
conv2d_42/ReluReluconv2d_42/BiasAdd:output:0*
T0*/
_output_shapes
:         <e
spatial_dropout2d_7/ShapeShapeconv2d_45/Relu:activations:0*
T0*
_output_shapes
:q
'spatial_dropout2d_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)spatial_dropout2d_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)spatial_dropout2d_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
!spatial_dropout2d_7/strided_sliceStridedSlice"spatial_dropout2d_7/Shape:output:00spatial_dropout2d_7/strided_slice/stack:output:02spatial_dropout2d_7/strided_slice/stack_1:output:02spatial_dropout2d_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
)spatial_dropout2d_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+spatial_dropout2d_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+spatial_dropout2d_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
#spatial_dropout2d_7/strided_slice_1StridedSlice"spatial_dropout2d_7/Shape:output:02spatial_dropout2d_7/strided_slice_1/stack:output:04spatial_dropout2d_7/strided_slice_1/stack_1:output:04spatial_dropout2d_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
!spatial_dropout2d_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?ф
spatial_dropout2d_7/dropout/MulMulconv2d_45/Relu:activations:0*spatial_dropout2d_7/dropout/Const:output:0*
T0*/
_output_shapes
:         t
2spatial_dropout2d_7/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :t
2spatial_dropout2d_7/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :║
0spatial_dropout2d_7/dropout/random_uniform/shapePack*spatial_dropout2d_7/strided_slice:output:0;spatial_dropout2d_7/dropout/random_uniform/shape/1:output:0;spatial_dropout2d_7/dropout/random_uniform/shape/2:output:0,spatial_dropout2d_7/strided_slice_1:output:0*
N*
T0*
_output_shapes
:╦
8spatial_dropout2d_7/dropout/random_uniform/RandomUniformRandomUniform9spatial_dropout2d_7/dropout/random_uniform/shape:output:0*
T0*/
_output_shapes
:         *
dtype0o
*spatial_dropout2d_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Ж
(spatial_dropout2d_7/dropout/GreaterEqualGreaterEqualAspatial_dropout2d_7/dropout/random_uniform/RandomUniform:output:03spatial_dropout2d_7/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         Ъ
 spatial_dropout2d_7/dropout/CastCast,spatial_dropout2d_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         Г
!spatial_dropout2d_7/dropout/Mul_1Mul#spatial_dropout2d_7/dropout/Mul:z:0$spatial_dropout2d_7/dropout/Cast:y:0*
T0*/
_output_shapes
:         ј
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype0њ
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype0░
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0┤
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_42/Relu:activations:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         <:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<ъ
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(е
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(љ
conv2d_46/Conv2D/ReadVariableOpReadVariableOp(conv2d_46_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╠
conv2d_46/Conv2DConv2D%spatial_dropout2d_7/dropout/Mul_1:z:0'conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
є
 conv2d_46/BiasAdd/ReadVariableOpReadVariableOp)conv2d_46_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Џ
conv2d_46/BiasAddBiasAddconv2d_46/Conv2D:output:0(conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         l
conv2d_46/ReluReluconv2d_46/BiasAdd:output:0*
T0*/
_output_shapes
:         љ
conv2d_43/Conv2D/ReadVariableOpReadVariableOp(conv2d_43_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0Л
conv2d_43/Conv2DConv2D*batch_normalization_7/FusedBatchNormV3:y:0'conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <*
paddingSAME*
strides
є
 conv2d_43/BiasAdd/ReadVariableOpReadVariableOp)conv2d_43_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Џ
conv2d_43/BiasAddBiasAddconv2d_43/Conv2D:output:0(conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <l
conv2d_43/ReluReluconv2d_43/BiasAdd:output:0*
T0*/
_output_shapes
:         <љ
conv2d_47/Conv2D/ReadVariableOpReadVariableOp(conv2d_47_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0├
conv2d_47/Conv2DConv2Dconv2d_46/Relu:activations:0'conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
є
 conv2d_47/BiasAdd/ReadVariableOpReadVariableOp)conv2d_47_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Џ
conv2d_47/BiasAddBiasAddconv2d_47/Conv2D:output:0(conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         l
conv2d_47/ReluReluconv2d_47/BiasAdd:output:0*
T0*/
_output_shapes
:         љ
conv2d_44/Conv2D/ReadVariableOpReadVariableOp(conv2d_44_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0├
conv2d_44/Conv2DConv2Dconv2d_43/Relu:activations:0'conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <*
paddingSAME*
strides
є
 conv2d_44/BiasAdd/ReadVariableOpReadVariableOp)conv2d_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Џ
conv2d_44/BiasAddBiasAddconv2d_44/Conv2D:output:0(conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <l
conv2d_44/ReluReluconv2d_44/BiasAdd:output:0*
T0*/
_output_shapes
:         <}
,global_max_pooling2d_7/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      е
global_max_pooling2d_7/MaxMaxconv2d_47/Relu:activations:05global_max_pooling2d_7/Max/reduction_indices:output:0*
T0*'
_output_shapes
:         a
flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ј
flatten_15/ReshapeReshape#global_max_pooling2d_7/Max:output:0flatten_15/Const:output:0*
T0*'
_output_shapes
:         a
flatten_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"    x   ѕ
flatten_14/ReshapeReshapeconv2d_44/Relu:activations:0flatten_14/Const:output:0*
T0*'
_output_shapes
:         x[
concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :╣
concatenate_7/concatConcatV2flatten_15/Reshape:output:0flatten_14/Reshape:output:0"concatenate_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:         zє
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

:z*
dtype0њ
dense_21/MatMulMatMulconcatenate_7/concat:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         b
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*'
_output_shapes
:         є
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:*
dtype0љ
dense_22/MatMulMatMuldense_21/Relu:activations:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         b
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*'
_output_shapes
:         є
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:*
dtype0љ
dense_23/MatMulMatMuldense_22/Relu:activations:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         b
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*'
_output_shapes
:         ю
)prediction_output_1/MatMul/ReadVariableOpReadVariableOp2prediction_output_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0д
prediction_output_1/MatMulMatMuldense_23/Relu:activations:01prediction_output_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         џ
*prediction_output_1/BiasAdd/ReadVariableOpReadVariableOp3prediction_output_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0▓
prediction_output_1/BiasAddBiasAdd$prediction_output_1/MatMul:product:02prediction_output_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ю
)prediction_output_0/MatMul/ReadVariableOpReadVariableOp2prediction_output_0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0д
prediction_output_0/MatMulMatMuldense_23/Relu:activations:01prediction_output_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         џ
*prediction_output_0/BiasAdd/ReadVariableOpReadVariableOp3prediction_output_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0▓
prediction_output_0/BiasAddBiasAdd$prediction_output_0/MatMul:product:02prediction_output_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         s
IdentityIdentity$prediction_output_0/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         u

Identity_1Identity$prediction_output_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ы
NoOpNoOp%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1!^conv2d_42/BiasAdd/ReadVariableOp ^conv2d_42/Conv2D/ReadVariableOp!^conv2d_43/BiasAdd/ReadVariableOp ^conv2d_43/Conv2D/ReadVariableOp!^conv2d_44/BiasAdd/ReadVariableOp ^conv2d_44/Conv2D/ReadVariableOp!^conv2d_45/BiasAdd/ReadVariableOp ^conv2d_45/Conv2D/ReadVariableOp!^conv2d_46/BiasAdd/ReadVariableOp ^conv2d_46/Conv2D/ReadVariableOp!^conv2d_47/BiasAdd/ReadVariableOp ^conv2d_47/Conv2D/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp+^prediction_output_0/BiasAdd/ReadVariableOp*^prediction_output_0/MatMul/ReadVariableOp+^prediction_output_1/BiasAdd/ReadVariableOp*^prediction_output_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:         <:         <<: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12D
 conv2d_42/BiasAdd/ReadVariableOp conv2d_42/BiasAdd/ReadVariableOp2B
conv2d_42/Conv2D/ReadVariableOpconv2d_42/Conv2D/ReadVariableOp2D
 conv2d_43/BiasAdd/ReadVariableOp conv2d_43/BiasAdd/ReadVariableOp2B
conv2d_43/Conv2D/ReadVariableOpconv2d_43/Conv2D/ReadVariableOp2D
 conv2d_44/BiasAdd/ReadVariableOp conv2d_44/BiasAdd/ReadVariableOp2B
conv2d_44/Conv2D/ReadVariableOpconv2d_44/Conv2D/ReadVariableOp2D
 conv2d_45/BiasAdd/ReadVariableOp conv2d_45/BiasAdd/ReadVariableOp2B
conv2d_45/Conv2D/ReadVariableOpconv2d_45/Conv2D/ReadVariableOp2D
 conv2d_46/BiasAdd/ReadVariableOp conv2d_46/BiasAdd/ReadVariableOp2B
conv2d_46/Conv2D/ReadVariableOpconv2d_46/Conv2D/ReadVariableOp2D
 conv2d_47/BiasAdd/ReadVariableOp conv2d_47/BiasAdd/ReadVariableOp2B
conv2d_47/Conv2D/ReadVariableOpconv2d_47/Conv2D/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2X
*prediction_output_0/BiasAdd/ReadVariableOp*prediction_output_0/BiasAdd/ReadVariableOp2V
)prediction_output_0/MatMul/ReadVariableOp)prediction_output_0/MatMul/ReadVariableOp2X
*prediction_output_1/BiasAdd/ReadVariableOp*prediction_output_1/BiasAdd/ReadVariableOp2V
)prediction_output_1/MatMul/ReadVariableOp)prediction_output_1/MatMul/ReadVariableOp:Y U
/
_output_shapes
:         <
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         <<
"
_user_specified_name
inputs/1
╠
ю
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_717837

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Я
Э
,__inference_joint_model_layer_call_fn_717409
inputs_0
inputs_1!
unknown:		
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:	

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:z

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:
identity

identity_1ѕбStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         :         *<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_joint_model_layer_call_and_return_conditional_losses_716690o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:         <:         <<: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:         <
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         <<
"
_user_specified_name
inputs/1
┼
ќ
)__inference_dense_23_layer_call_fn_718010

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_716650o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ё
■
E__inference_conv2d_44_layer_call_and_return_conditional_losses_716573

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         <i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         <w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         <: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         <
 
_user_specified_nameinputs
їн
х6
"__inference__traced_restore_718604
file_prefix;
!assignvariableop_conv2d_45_kernel:		/
!assignvariableop_1_conv2d_45_bias:=
#assignvariableop_2_conv2d_42_kernel:/
!assignvariableop_3_conv2d_42_bias:=
#assignvariableop_4_conv2d_46_kernel:/
!assignvariableop_5_conv2d_46_bias:<
.assignvariableop_6_batch_normalization_7_gamma:;
-assignvariableop_7_batch_normalization_7_beta:B
4assignvariableop_8_batch_normalization_7_moving_mean:F
8assignvariableop_9_batch_normalization_7_moving_variance:>
$assignvariableop_10_conv2d_47_kernel:0
"assignvariableop_11_conv2d_47_bias:>
$assignvariableop_12_conv2d_43_kernel:	0
"assignvariableop_13_conv2d_43_bias:>
$assignvariableop_14_conv2d_44_kernel:0
"assignvariableop_15_conv2d_44_bias:5
#assignvariableop_16_dense_21_kernel:z/
!assignvariableop_17_dense_21_bias:5
#assignvariableop_18_dense_22_kernel:/
!assignvariableop_19_dense_22_bias:5
#assignvariableop_20_dense_23_kernel:/
!assignvariableop_21_dense_23_bias:@
.assignvariableop_22_prediction_output_0_kernel::
,assignvariableop_23_prediction_output_0_bias:@
.assignvariableop_24_prediction_output_1_kernel::
,assignvariableop_25_prediction_output_1_bias:'
assignvariableop_26_adam_iter:	 )
assignvariableop_27_adam_beta_1: )
assignvariableop_28_adam_beta_2: (
assignvariableop_29_adam_decay: 0
&assignvariableop_30_adam_learning_rate: %
assignvariableop_31_total_2: %
assignvariableop_32_count_2: %
assignvariableop_33_total_1: %
assignvariableop_34_count_1: #
assignvariableop_35_total: #
assignvariableop_36_count: E
+assignvariableop_37_adam_conv2d_45_kernel_m:		7
)assignvariableop_38_adam_conv2d_45_bias_m:E
+assignvariableop_39_adam_conv2d_42_kernel_m:7
)assignvariableop_40_adam_conv2d_42_bias_m:E
+assignvariableop_41_adam_conv2d_46_kernel_m:7
)assignvariableop_42_adam_conv2d_46_bias_m:D
6assignvariableop_43_adam_batch_normalization_7_gamma_m:C
5assignvariableop_44_adam_batch_normalization_7_beta_m:E
+assignvariableop_45_adam_conv2d_47_kernel_m:7
)assignvariableop_46_adam_conv2d_47_bias_m:E
+assignvariableop_47_adam_conv2d_43_kernel_m:	7
)assignvariableop_48_adam_conv2d_43_bias_m:E
+assignvariableop_49_adam_conv2d_44_kernel_m:7
)assignvariableop_50_adam_conv2d_44_bias_m:<
*assignvariableop_51_adam_dense_21_kernel_m:z6
(assignvariableop_52_adam_dense_21_bias_m:<
*assignvariableop_53_adam_dense_22_kernel_m:6
(assignvariableop_54_adam_dense_22_bias_m:<
*assignvariableop_55_adam_dense_23_kernel_m:6
(assignvariableop_56_adam_dense_23_bias_m:G
5assignvariableop_57_adam_prediction_output_0_kernel_m:A
3assignvariableop_58_adam_prediction_output_0_bias_m:G
5assignvariableop_59_adam_prediction_output_1_kernel_m:A
3assignvariableop_60_adam_prediction_output_1_bias_m:E
+assignvariableop_61_adam_conv2d_45_kernel_v:		7
)assignvariableop_62_adam_conv2d_45_bias_v:E
+assignvariableop_63_adam_conv2d_42_kernel_v:7
)assignvariableop_64_adam_conv2d_42_bias_v:E
+assignvariableop_65_adam_conv2d_46_kernel_v:7
)assignvariableop_66_adam_conv2d_46_bias_v:D
6assignvariableop_67_adam_batch_normalization_7_gamma_v:C
5assignvariableop_68_adam_batch_normalization_7_beta_v:E
+assignvariableop_69_adam_conv2d_47_kernel_v:7
)assignvariableop_70_adam_conv2d_47_bias_v:E
+assignvariableop_71_adam_conv2d_43_kernel_v:	7
)assignvariableop_72_adam_conv2d_43_bias_v:E
+assignvariableop_73_adam_conv2d_44_kernel_v:7
)assignvariableop_74_adam_conv2d_44_bias_v:<
*assignvariableop_75_adam_dense_21_kernel_v:z6
(assignvariableop_76_adam_dense_21_bias_v:<
*assignvariableop_77_adam_dense_22_kernel_v:6
(assignvariableop_78_adam_dense_22_bias_v:<
*assignvariableop_79_adam_dense_23_kernel_v:6
(assignvariableop_80_adam_dense_23_bias_v:G
5assignvariableop_81_adam_prediction_output_0_kernel_v:A
3assignvariableop_82_adam_prediction_output_0_bias_v:G
5assignvariableop_83_adam_prediction_output_1_kernel_v:A
3assignvariableop_84_adam_prediction_output_1_bias_v:
identity_86ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_48бAssignVariableOp_49бAssignVariableOp_5бAssignVariableOp_50бAssignVariableOp_51бAssignVariableOp_52бAssignVariableOp_53бAssignVariableOp_54бAssignVariableOp_55бAssignVariableOp_56бAssignVariableOp_57бAssignVariableOp_58бAssignVariableOp_59бAssignVariableOp_6бAssignVariableOp_60бAssignVariableOp_61бAssignVariableOp_62бAssignVariableOp_63бAssignVariableOp_64бAssignVariableOp_65бAssignVariableOp_66бAssignVariableOp_67бAssignVariableOp_68бAssignVariableOp_69бAssignVariableOp_7бAssignVariableOp_70бAssignVariableOp_71бAssignVariableOp_72бAssignVariableOp_73бAssignVariableOp_74бAssignVariableOp_75бAssignVariableOp_76бAssignVariableOp_77бAssignVariableOp_78бAssignVariableOp_79бAssignVariableOp_8бAssignVariableOp_80бAssignVariableOp_81бAssignVariableOp_82бAssignVariableOp_83бAssignVariableOp_84бAssignVariableOp_9ш/
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:V*
dtype0*Џ/
valueЉ/Bј/VB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЪ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:V*
dtype0*┴
valueиB┤VB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ¤
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ь
_output_shapes█
п::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*d
dtypesZ
X2V	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_45_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_45_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_42_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_42_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_46_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_46_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_6AssignVariableOp.assignvariableop_6_batch_normalization_7_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_7AssignVariableOp-assignvariableop_7_batch_normalization_7_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_8AssignVariableOp4assignvariableop_8_batch_normalization_7_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_9AssignVariableOp8assignvariableop_9_batch_normalization_7_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_47_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_47_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_43_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_43_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_14AssignVariableOp$assignvariableop_14_conv2d_44_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_15AssignVariableOp"assignvariableop_15_conv2d_44_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:ћ
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_21_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_21_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:ћ
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_22_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_22_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:ћ
AssignVariableOp_20AssignVariableOp#assignvariableop_20_dense_23_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_21AssignVariableOp!assignvariableop_21_dense_23_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_22AssignVariableOp.assignvariableop_22_prediction_output_0_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_23AssignVariableOp,assignvariableop_23_prediction_output_0_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_24AssignVariableOp.assignvariableop_24_prediction_output_1_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_25AssignVariableOp,assignvariableop_25_prediction_output_1_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0	*
_output_shapes
:ј
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_iterIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_27AssignVariableOpassignvariableop_27_adam_beta_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_28AssignVariableOpassignvariableop_28_adam_beta_2Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_29AssignVariableOpassignvariableop_29_adam_decayIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_learning_rateIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_31AssignVariableOpassignvariableop_31_total_2Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_32AssignVariableOpassignvariableop_32_count_2Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_33AssignVariableOpassignvariableop_33_total_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_34AssignVariableOpassignvariableop_34_count_1Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_35AssignVariableOpassignvariableop_35_totalIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_36AssignVariableOpassignvariableop_36_countIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_45_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_45_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_42_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_42_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv2d_46_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv2d_46_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_43AssignVariableOp6assignvariableop_43_adam_batch_normalization_7_gamma_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_44AssignVariableOp5assignvariableop_44_adam_batch_normalization_7_beta_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_47_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_47_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_43_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_43_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_conv2d_44_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_conv2d_44_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_21_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_21_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_dense_22_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense_22_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_dense_23_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_dense_23_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_57AssignVariableOp5assignvariableop_57_adam_prediction_output_0_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_58AssignVariableOp3assignvariableop_58_adam_prediction_output_0_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_59AssignVariableOp5assignvariableop_59_adam_prediction_output_1_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_60AssignVariableOp3assignvariableop_60_adam_prediction_output_1_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_61AssignVariableOp+assignvariableop_61_adam_conv2d_45_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_62AssignVariableOp)assignvariableop_62_adam_conv2d_45_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adam_conv2d_42_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adam_conv2d_42_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_conv2d_46_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_conv2d_46_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_67AssignVariableOp6assignvariableop_67_adam_batch_normalization_7_gamma_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_68AssignVariableOp5assignvariableop_68_adam_batch_normalization_7_beta_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_conv2d_47_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_conv2d_47_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_conv2d_43_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_conv2d_43_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_73AssignVariableOp+assignvariableop_73_adam_conv2d_44_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_74AssignVariableOp)assignvariableop_74_adam_conv2d_44_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_75AssignVariableOp*assignvariableop_75_adam_dense_21_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_76AssignVariableOp(assignvariableop_76_adam_dense_21_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_dense_22_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_dense_22_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_79AssignVariableOp*assignvariableop_79_adam_dense_23_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_80AssignVariableOp(assignvariableop_80_adam_dense_23_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_81AssignVariableOp5assignvariableop_81_adam_prediction_output_0_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_82AssignVariableOp3assignvariableop_82_adam_prediction_output_0_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_83AssignVariableOp5assignvariableop_83_adam_prediction_output_1_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_84AssignVariableOp3assignvariableop_84_adam_prediction_output_1_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ю
Identity_85Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_86IdentityIdentity_85:output:0^NoOp_1*
T0*
_output_shapes
: і
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_86Identity_86:output:0*┴
_input_shapes»
г: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
№
Ъ
*__inference_conv2d_47_layer_call_fn_717864

inputs!
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_47_layer_call_and_return_conditional_losses_716556w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
■
n
O__inference_spatial_dropout2d_7_layer_call_and_return_conditional_losses_717753

inputs
identityѕ;
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
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
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
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?Є
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4                                    `
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :`
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :о
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:г
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"                  *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>и
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"                  ђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"                  ї
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4                                    |
IdentityIdentitydropout/Mul_1:z:0*
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
Л
6__inference_batch_normalization_7_layer_call_fn_717806

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_716403Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
рU
№
G__inference_joint_model_layer_call_and_return_conditional_losses_717281
input_15
input_16*
conv2d_45_717210:		
conv2d_45_717212:*
conv2d_42_717215:
conv2d_42_717217:*
batch_normalization_7_717221:*
batch_normalization_7_717223:*
batch_normalization_7_717225:*
batch_normalization_7_717227:*
conv2d_46_717230:
conv2d_46_717232:*
conv2d_43_717235:	
conv2d_43_717237:*
conv2d_47_717240:
conv2d_47_717242:*
conv2d_44_717245:
conv2d_44_717247:!
dense_21_717254:z
dense_21_717256:!
dense_22_717259:
dense_22_717261:!
dense_23_717264:
dense_23_717266:,
prediction_output_1_717269:(
prediction_output_1_717271:,
prediction_output_0_717274:(
prediction_output_0_717276:
identity

identity_1ѕб-batch_normalization_7/StatefulPartitionedCallб!conv2d_42/StatefulPartitionedCallб!conv2d_43/StatefulPartitionedCallб!conv2d_44/StatefulPartitionedCallб!conv2d_45/StatefulPartitionedCallб!conv2d_46/StatefulPartitionedCallб!conv2d_47/StatefulPartitionedCallб dense_21/StatefulPartitionedCallб dense_22/StatefulPartitionedCallб dense_23/StatefulPartitionedCallб+prediction_output_0/StatefulPartitionedCallб+prediction_output_1/StatefulPartitionedCallб+spatial_dropout2d_7/StatefulPartitionedCallЂ
!conv2d_45/StatefulPartitionedCallStatefulPartitionedCallinput_16conv2d_45_717210conv2d_45_717212*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_45_layer_call_and_return_conditional_losses_716478Ђ
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCallinput_15conv2d_42_717215conv2d_42_717217*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         <*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_42_layer_call_and_return_conditional_losses_716495Ї
+spatial_dropout2d_7/StatefulPartitionedCallStatefulPartitionedCall*conv2d_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_spatial_dropout2d_7_layer_call_and_return_conditional_losses_716378Љ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0batch_normalization_7_717221batch_normalization_7_717223batch_normalization_7_717225batch_normalization_7_717227*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         <*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_716434Г
!conv2d_46/StatefulPartitionedCallStatefulPartitionedCall4spatial_dropout2d_7/StatefulPartitionedCall:output:0conv2d_46_717230conv2d_46_717232*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_46_layer_call_and_return_conditional_losses_716522»
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv2d_43_717235conv2d_43_717237*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         <*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_43_layer_call_and_return_conditional_losses_716539Б
!conv2d_47/StatefulPartitionedCallStatefulPartitionedCall*conv2d_46/StatefulPartitionedCall:output:0conv2d_47_717240conv2d_47_717242*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_47_layer_call_and_return_conditional_losses_716556Б
!conv2d_44/StatefulPartitionedCallStatefulPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0conv2d_44_717245conv2d_44_717247*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         <*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_44_layer_call_and_return_conditional_losses_716573ч
&global_max_pooling2d_7/PartitionedCallPartitionedCall*conv2d_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_global_max_pooling2d_7_layer_call_and_return_conditional_losses_716455У
flatten_15/PartitionedCallPartitionedCall/global_max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_flatten_15_layer_call_and_return_conditional_losses_716586с
flatten_14/PartitionedCallPartitionedCall*conv2d_44/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         x* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_flatten_14_layer_call_and_return_conditional_losses_716594ѕ
concatenate_7/PartitionedCallPartitionedCall#flatten_15/PartitionedCall:output:0#flatten_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         z* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_concatenate_7_layer_call_and_return_conditional_losses_716603Њ
 dense_21/StatefulPartitionedCallStatefulPartitionedCall&concatenate_7/PartitionedCall:output:0dense_21_717254dense_21_717256*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_716616ќ
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_717259dense_22_717261*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_716633ќ
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_717264dense_23_717266*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_716650┬
+prediction_output_1/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0prediction_output_1_717269prediction_output_1_717271*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_prediction_output_1_layer_call_and_return_conditional_losses_716666┬
+prediction_output_0/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0prediction_output_0_717274prediction_output_0_717276*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_prediction_output_0_layer_call_and_return_conditional_losses_716682Ѓ
IdentityIdentity4prediction_output_0/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Ё

Identity_1Identity4prediction_output_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ┴
NoOpNoOp.^batch_normalization_7/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall"^conv2d_44/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall,^prediction_output_0/StatefulPartitionedCall,^prediction_output_1/StatefulPartitionedCall,^spatial_dropout2d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:         <:         <<: : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2F
!conv2d_44/StatefulPartitionedCall!conv2d_44/StatefulPartitionedCall2F
!conv2d_45/StatefulPartitionedCall!conv2d_45/StatefulPartitionedCall2F
!conv2d_46/StatefulPartitionedCall!conv2d_46/StatefulPartitionedCall2F
!conv2d_47/StatefulPartitionedCall!conv2d_47/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2Z
+prediction_output_0/StatefulPartitionedCall+prediction_output_0/StatefulPartitionedCall2Z
+prediction_output_1/StatefulPartitionedCall+prediction_output_1/StatefulPartitionedCall2Z
+spatial_dropout2d_7/StatefulPartitionedCall+spatial_dropout2d_7/StatefulPartitionedCall:Y U
/
_output_shapes
:         <
"
_user_specified_name
input_15:YU
/
_output_shapes
:         <<
"
_user_specified_name
input_16
ё
■
E__inference_conv2d_43_layer_call_and_return_conditional_losses_716539

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         <i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         <w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         <: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         <
 
_user_specified_nameinputs
Ь
m
O__inference_spatial_dropout2d_7_layer_call_and_return_conditional_losses_716350

inputs

identity_1q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4                                    ~

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4                                    "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
є
└
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_716434

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
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
-:+                           н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
м	
ђ
O__inference_prediction_output_1_layer_call_and_return_conditional_losses_716666

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
к
b
F__inference_flatten_14_layer_call_and_return_conditional_losses_716594

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    x   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         xX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         x"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         <:W S
/
_output_shapes
:         <
 
_user_specified_nameinputs
ё
■
E__inference_conv2d_45_layer_call_and_return_conditional_losses_716478

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         <<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         <<
 
_user_specified_nameinputs
Ћ
m
4__inference_spatial_dropout2d_7_layer_call_fn_717725

inputs
identityѕбStatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputs*
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
GPU2*0J 8ѓ *X
fSRQ
O__inference_spatial_dropout2d_7_layer_call_and_return_conditional_losses_716378њ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*J
_output_shapes8
6:4                                    `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Џ

ш
D__inference_dense_22_layer_call_and_return_conditional_losses_718001

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
я
Э
,__inference_joint_model_layer_call_fn_717131
input_15
input_16!
unknown:		
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:	

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:z

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:
identity

identity_1ѕбStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinput_15input_16unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         :         *:
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_joint_model_layer_call_and_return_conditional_losses_717014o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:         <:         <<: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:         <
"
_user_specified_name
input_15:YU
/
_output_shapes
:         <<
"
_user_specified_name
input_16
█
А
4__inference_prediction_output_1_layer_call_fn_718049

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_prediction_output_1_layer_call_and_return_conditional_losses_716666o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Џ

ш
D__inference_dense_23_layer_call_and_return_conditional_losses_718021

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
№
Ъ
*__inference_conv2d_42_layer_call_fn_717762

inputs!
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         <*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_42_layer_call_and_return_conditional_losses_716495w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         <`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         <: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         <
 
_user_specified_nameinputs
ё
■
E__inference_conv2d_45_layer_call_and_return_conditional_losses_717715

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         <<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         <<
 
_user_specified_nameinputs
┤
G
+__inference_flatten_14_layer_call_fn_717942

inputs
identity┤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         x* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_flatten_14_layer_call_and_return_conditional_losses_716594`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         x"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         <:W S
/
_output_shapes
:         <
 
_user_specified_nameinputs
ћ
S
7__inference_global_max_pooling2d_7_layer_call_fn_717900

inputs
identity╔
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:                  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_global_max_pooling2d_7_layer_call_and_return_conditional_losses_716455i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
м	
ђ
O__inference_prediction_output_0_layer_call_and_return_conditional_losses_716682

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
├
P
4__inference_spatial_dropout2d_7_layer_call_fn_717720

inputs
identityЯ
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
GPU2*0J 8ѓ *X
fSRQ
O__inference_spatial_dropout2d_7_layer_call_and_return_conditional_losses_716350Ѓ
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
ё
■
E__inference_conv2d_46_layer_call_and_return_conditional_losses_716522

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Х
b
F__inference_flatten_15_layer_call_and_return_conditional_losses_717937

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Џ

ш
D__inference_dense_21_layer_call_and_return_conditional_losses_716616

inputs0
matmul_readvariableop_resource:z-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:z*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         z
 
_user_specified_nameinputs
Х
b
F__inference_flatten_15_layer_call_and_return_conditional_losses_716586

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ё
■
E__inference_conv2d_47_layer_call_and_return_conditional_losses_716556

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
▓
­
$__inference_signature_wrapper_717349
input_15
input_16!
unknown:		
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:	

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:z

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:
identity

identity_1ѕбStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinput_15input_16unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         :         *<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ **
f%R#
!__inference__wrapped_model_716341o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:         <:         <<: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:         <
"
_user_specified_name
input_15:YU
/
_output_shapes
:         <<
"
_user_specified_name
input_16
Я
Э
,__inference_joint_model_layer_call_fn_716747
input_15
input_16!
unknown:		
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:	

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:z

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:
identity

identity_1ѕбStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinput_15input_16unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         :         *<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_joint_model_layer_call_and_return_conditional_losses_716690o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:         <:         <<: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:         <
"
_user_specified_name
input_15:YU
/
_output_shapes
:         <<
"
_user_specified_name
input_16
ц
G
+__inference_flatten_15_layer_call_fn_717931

inputs
identity┤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_flatten_15_layer_call_and_return_conditional_losses_716586`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
№
Ъ
*__inference_conv2d_45_layer_call_fn_717704

inputs!
unknown:		
	unknown_0:
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_45_layer_call_and_return_conditional_losses_716478w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         <<: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         <<
 
_user_specified_nameinputs
Ь
m
O__inference_spatial_dropout2d_7_layer_call_and_return_conditional_losses_717730

inputs

identity_1q
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4                                    ~

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4                                    "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ё
■
E__inference_conv2d_44_layer_call_and_return_conditional_losses_717926

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         <i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         <w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         <: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         <
 
_user_specified_nameinputs
ё
■
E__inference_conv2d_43_layer_call_and_return_conditional_losses_717895

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         <i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         <w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         <: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         <
 
_user_specified_nameinputs
м	
ђ
O__inference_prediction_output_0_layer_call_and_return_conditional_losses_718040

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
к
b
F__inference_flatten_14_layer_call_and_return_conditional_losses_717948

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    x   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         xX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         x"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         <:W S
/
_output_shapes
:         <
 
_user_specified_nameinputs
«
n
R__inference_global_max_pooling2d_7_layer_call_and_return_conditional_losses_716455

inputs
identityf
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:                  ]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Џ

ш
D__inference_dense_22_layer_call_and_return_conditional_losses_716633

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Џ

ш
D__inference_dense_23_layer_call_and_return_conditional_losses_716650

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ЊT
┴
G__inference_joint_model_layer_call_and_return_conditional_losses_717206
input_15
input_16*
conv2d_45_717135:		
conv2d_45_717137:*
conv2d_42_717140:
conv2d_42_717142:*
batch_normalization_7_717146:*
batch_normalization_7_717148:*
batch_normalization_7_717150:*
batch_normalization_7_717152:*
conv2d_46_717155:
conv2d_46_717157:*
conv2d_43_717160:	
conv2d_43_717162:*
conv2d_47_717165:
conv2d_47_717167:*
conv2d_44_717170:
conv2d_44_717172:!
dense_21_717179:z
dense_21_717181:!
dense_22_717184:
dense_22_717186:!
dense_23_717189:
dense_23_717191:,
prediction_output_1_717194:(
prediction_output_1_717196:,
prediction_output_0_717199:(
prediction_output_0_717201:
identity

identity_1ѕб-batch_normalization_7/StatefulPartitionedCallб!conv2d_42/StatefulPartitionedCallб!conv2d_43/StatefulPartitionedCallб!conv2d_44/StatefulPartitionedCallб!conv2d_45/StatefulPartitionedCallб!conv2d_46/StatefulPartitionedCallб!conv2d_47/StatefulPartitionedCallб dense_21/StatefulPartitionedCallб dense_22/StatefulPartitionedCallб dense_23/StatefulPartitionedCallб+prediction_output_0/StatefulPartitionedCallб+prediction_output_1/StatefulPartitionedCallЂ
!conv2d_45/StatefulPartitionedCallStatefulPartitionedCallinput_16conv2d_45_717135conv2d_45_717137*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_45_layer_call_and_return_conditional_losses_716478Ђ
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCallinput_15conv2d_42_717140conv2d_42_717142*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         <*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_42_layer_call_and_return_conditional_losses_716495§
#spatial_dropout2d_7/PartitionedCallPartitionedCall*conv2d_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_spatial_dropout2d_7_layer_call_and_return_conditional_losses_716350Њ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0batch_normalization_7_717146batch_normalization_7_717148batch_normalization_7_717150batch_normalization_7_717152*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         <*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_716403Ц
!conv2d_46/StatefulPartitionedCallStatefulPartitionedCall,spatial_dropout2d_7/PartitionedCall:output:0conv2d_46_717155conv2d_46_717157*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_46_layer_call_and_return_conditional_losses_716522»
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv2d_43_717160conv2d_43_717162*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         <*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_43_layer_call_and_return_conditional_losses_716539Б
!conv2d_47/StatefulPartitionedCallStatefulPartitionedCall*conv2d_46/StatefulPartitionedCall:output:0conv2d_47_717165conv2d_47_717167*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_47_layer_call_and_return_conditional_losses_716556Б
!conv2d_44/StatefulPartitionedCallStatefulPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0conv2d_44_717170conv2d_44_717172*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         <*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_44_layer_call_and_return_conditional_losses_716573ч
&global_max_pooling2d_7/PartitionedCallPartitionedCall*conv2d_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_global_max_pooling2d_7_layer_call_and_return_conditional_losses_716455У
flatten_15/PartitionedCallPartitionedCall/global_max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_flatten_15_layer_call_and_return_conditional_losses_716586с
flatten_14/PartitionedCallPartitionedCall*conv2d_44/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         x* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_flatten_14_layer_call_and_return_conditional_losses_716594ѕ
concatenate_7/PartitionedCallPartitionedCall#flatten_15/PartitionedCall:output:0#flatten_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         z* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_concatenate_7_layer_call_and_return_conditional_losses_716603Њ
 dense_21/StatefulPartitionedCallStatefulPartitionedCall&concatenate_7/PartitionedCall:output:0dense_21_717179dense_21_717181*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_716616ќ
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_717184dense_22_717186*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_716633ќ
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_717189dense_23_717191*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_716650┬
+prediction_output_1/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0prediction_output_1_717194prediction_output_1_717196*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_prediction_output_1_layer_call_and_return_conditional_losses_716666┬
+prediction_output_0/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0prediction_output_0_717199prediction_output_0_717201*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_prediction_output_0_layer_call_and_return_conditional_losses_716682Ѓ
IdentityIdentity4prediction_output_0/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Ё

Identity_1Identity4prediction_output_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Њ
NoOpNoOp.^batch_normalization_7/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall"^conv2d_44/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall,^prediction_output_0/StatefulPartitionedCall,^prediction_output_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:         <:         <<: : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2F
!conv2d_44/StatefulPartitionedCall!conv2d_44/StatefulPartitionedCall2F
!conv2d_45/StatefulPartitionedCall!conv2d_45/StatefulPartitionedCall2F
!conv2d_46/StatefulPartitionedCall!conv2d_46/StatefulPartitionedCall2F
!conv2d_47/StatefulPartitionedCall!conv2d_47/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2Z
+prediction_output_0/StatefulPartitionedCall+prediction_output_0/StatefulPartitionedCall2Z
+prediction_output_1/StatefulPartitionedCall+prediction_output_1/StatefulPartitionedCall:Y U
/
_output_shapes
:         <
"
_user_specified_name
input_15:YU
/
_output_shapes
:         <<
"
_user_specified_name
input_16
ё
■
E__inference_conv2d_42_layer_call_and_return_conditional_losses_716495

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         <i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         <w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         <: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         <
 
_user_specified_nameinputs
┼
ќ
)__inference_dense_21_layer_call_fn_717970

inputs
unknown:z
	unknown_0:
identityѕбStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_716616o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         z: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         z
 
_user_specified_nameinputs
║
s
I__inference_concatenate_7_layer_call_and_return_conditional_losses_716603

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:         zW
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:         z"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         :         x:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         x
 
_user_specified_nameinputs
№
Ъ
*__inference_conv2d_44_layer_call_fn_717915

inputs!
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         <*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_44_layer_call_and_return_conditional_losses_716573w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         <`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         <: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         <
 
_user_specified_nameinputs
╠
ю
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_716403

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
█
А
4__inference_prediction_output_0_layer_call_fn_718030

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_prediction_output_0_layer_call_and_return_conditional_losses_716682o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
«
n
R__inference_global_max_pooling2d_7_layer_call_and_return_conditional_losses_717906

inputs
identityf
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:                  ]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
»
Z
.__inference_concatenate_7_layer_call_fn_717954
inputs_0
inputs_1
identity─
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         z* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_concatenate_7_layer_call_and_return_conditional_losses_716603`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         z"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         :         x:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         x
"
_user_specified_name
inputs/1
┘U
ь
G__inference_joint_model_layer_call_and_return_conditional_losses_717014

inputs
inputs_1*
conv2d_45_716943:		
conv2d_45_716945:*
conv2d_42_716948:
conv2d_42_716950:*
batch_normalization_7_716954:*
batch_normalization_7_716956:*
batch_normalization_7_716958:*
batch_normalization_7_716960:*
conv2d_46_716963:
conv2d_46_716965:*
conv2d_43_716968:	
conv2d_43_716970:*
conv2d_47_716973:
conv2d_47_716975:*
conv2d_44_716978:
conv2d_44_716980:!
dense_21_716987:z
dense_21_716989:!
dense_22_716992:
dense_22_716994:!
dense_23_716997:
dense_23_716999:,
prediction_output_1_717002:(
prediction_output_1_717004:,
prediction_output_0_717007:(
prediction_output_0_717009:
identity

identity_1ѕб-batch_normalization_7/StatefulPartitionedCallб!conv2d_42/StatefulPartitionedCallб!conv2d_43/StatefulPartitionedCallб!conv2d_44/StatefulPartitionedCallб!conv2d_45/StatefulPartitionedCallб!conv2d_46/StatefulPartitionedCallб!conv2d_47/StatefulPartitionedCallб dense_21/StatefulPartitionedCallб dense_22/StatefulPartitionedCallб dense_23/StatefulPartitionedCallб+prediction_output_0/StatefulPartitionedCallб+prediction_output_1/StatefulPartitionedCallб+spatial_dropout2d_7/StatefulPartitionedCallЂ
!conv2d_45/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv2d_45_716943conv2d_45_716945*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_45_layer_call_and_return_conditional_losses_716478 
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_42_716948conv2d_42_716950*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         <*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_42_layer_call_and_return_conditional_losses_716495Ї
+spatial_dropout2d_7/StatefulPartitionedCallStatefulPartitionedCall*conv2d_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_spatial_dropout2d_7_layer_call_and_return_conditional_losses_716378Љ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0batch_normalization_7_716954batch_normalization_7_716956batch_normalization_7_716958batch_normalization_7_716960*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         <*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_716434Г
!conv2d_46/StatefulPartitionedCallStatefulPartitionedCall4spatial_dropout2d_7/StatefulPartitionedCall:output:0conv2d_46_716963conv2d_46_716965*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_46_layer_call_and_return_conditional_losses_716522»
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv2d_43_716968conv2d_43_716970*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         <*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_43_layer_call_and_return_conditional_losses_716539Б
!conv2d_47/StatefulPartitionedCallStatefulPartitionedCall*conv2d_46/StatefulPartitionedCall:output:0conv2d_47_716973conv2d_47_716975*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_47_layer_call_and_return_conditional_losses_716556Б
!conv2d_44/StatefulPartitionedCallStatefulPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0conv2d_44_716978conv2d_44_716980*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         <*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_44_layer_call_and_return_conditional_losses_716573ч
&global_max_pooling2d_7/PartitionedCallPartitionedCall*conv2d_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_global_max_pooling2d_7_layer_call_and_return_conditional_losses_716455У
flatten_15/PartitionedCallPartitionedCall/global_max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_flatten_15_layer_call_and_return_conditional_losses_716586с
flatten_14/PartitionedCallPartitionedCall*conv2d_44/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         x* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_flatten_14_layer_call_and_return_conditional_losses_716594ѕ
concatenate_7/PartitionedCallPartitionedCall#flatten_15/PartitionedCall:output:0#flatten_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         z* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_concatenate_7_layer_call_and_return_conditional_losses_716603Њ
 dense_21/StatefulPartitionedCallStatefulPartitionedCall&concatenate_7/PartitionedCall:output:0dense_21_716987dense_21_716989*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_716616ќ
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_716992dense_22_716994*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_716633ќ
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_716997dense_23_716999*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_716650┬
+prediction_output_1/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0prediction_output_1_717002prediction_output_1_717004*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_prediction_output_1_layer_call_and_return_conditional_losses_716666┬
+prediction_output_0/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0prediction_output_0_717007prediction_output_0_717009*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_prediction_output_0_layer_call_and_return_conditional_losses_716682Ѓ
IdentityIdentity4prediction_output_0/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Ё

Identity_1Identity4prediction_output_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ┴
NoOpNoOp.^batch_normalization_7/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall"^conv2d_44/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall,^prediction_output_0/StatefulPartitionedCall,^prediction_output_1/StatefulPartitionedCall,^spatial_dropout2d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:         <:         <<: : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2F
!conv2d_44/StatefulPartitionedCall!conv2d_44/StatefulPartitionedCall2F
!conv2d_45/StatefulPartitionedCall!conv2d_45/StatefulPartitionedCall2F
!conv2d_46/StatefulPartitionedCall!conv2d_46/StatefulPartitionedCall2F
!conv2d_47/StatefulPartitionedCall!conv2d_47/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2Z
+prediction_output_0/StatefulPartitionedCall+prediction_output_0/StatefulPartitionedCall2Z
+prediction_output_1/StatefulPartitionedCall+prediction_output_1/StatefulPartitionedCall2Z
+spatial_dropout2d_7/StatefulPartitionedCall+spatial_dropout2d_7/StatefulPartitionedCall:W S
/
_output_shapes
:         <
 
_user_specified_nameinputs:WS
/
_output_shapes
:         <<
 
_user_specified_nameinputs
кю
ъ
!__inference__wrapped_model_716341
input_15
input_16N
4joint_model_conv2d_45_conv2d_readvariableop_resource:		C
5joint_model_conv2d_45_biasadd_readvariableop_resource:N
4joint_model_conv2d_42_conv2d_readvariableop_resource:C
5joint_model_conv2d_42_biasadd_readvariableop_resource:G
9joint_model_batch_normalization_7_readvariableop_resource:I
;joint_model_batch_normalization_7_readvariableop_1_resource:X
Jjoint_model_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:Z
Ljoint_model_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:N
4joint_model_conv2d_46_conv2d_readvariableop_resource:C
5joint_model_conv2d_46_biasadd_readvariableop_resource:N
4joint_model_conv2d_43_conv2d_readvariableop_resource:	C
5joint_model_conv2d_43_biasadd_readvariableop_resource:N
4joint_model_conv2d_47_conv2d_readvariableop_resource:C
5joint_model_conv2d_47_biasadd_readvariableop_resource:N
4joint_model_conv2d_44_conv2d_readvariableop_resource:C
5joint_model_conv2d_44_biasadd_readvariableop_resource:E
3joint_model_dense_21_matmul_readvariableop_resource:zB
4joint_model_dense_21_biasadd_readvariableop_resource:E
3joint_model_dense_22_matmul_readvariableop_resource:B
4joint_model_dense_22_biasadd_readvariableop_resource:E
3joint_model_dense_23_matmul_readvariableop_resource:B
4joint_model_dense_23_biasadd_readvariableop_resource:P
>joint_model_prediction_output_1_matmul_readvariableop_resource:M
?joint_model_prediction_output_1_biasadd_readvariableop_resource:P
>joint_model_prediction_output_0_matmul_readvariableop_resource:M
?joint_model_prediction_output_0_biasadd_readvariableop_resource:
identity

identity_1ѕбAjoint_model/batch_normalization_7/FusedBatchNormV3/ReadVariableOpбCjoint_model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б0joint_model/batch_normalization_7/ReadVariableOpб2joint_model/batch_normalization_7/ReadVariableOp_1б,joint_model/conv2d_42/BiasAdd/ReadVariableOpб+joint_model/conv2d_42/Conv2D/ReadVariableOpб,joint_model/conv2d_43/BiasAdd/ReadVariableOpб+joint_model/conv2d_43/Conv2D/ReadVariableOpб,joint_model/conv2d_44/BiasAdd/ReadVariableOpб+joint_model/conv2d_44/Conv2D/ReadVariableOpб,joint_model/conv2d_45/BiasAdd/ReadVariableOpб+joint_model/conv2d_45/Conv2D/ReadVariableOpб,joint_model/conv2d_46/BiasAdd/ReadVariableOpб+joint_model/conv2d_46/Conv2D/ReadVariableOpб,joint_model/conv2d_47/BiasAdd/ReadVariableOpб+joint_model/conv2d_47/Conv2D/ReadVariableOpб+joint_model/dense_21/BiasAdd/ReadVariableOpб*joint_model/dense_21/MatMul/ReadVariableOpб+joint_model/dense_22/BiasAdd/ReadVariableOpб*joint_model/dense_22/MatMul/ReadVariableOpб+joint_model/dense_23/BiasAdd/ReadVariableOpб*joint_model/dense_23/MatMul/ReadVariableOpб6joint_model/prediction_output_0/BiasAdd/ReadVariableOpб5joint_model/prediction_output_0/MatMul/ReadVariableOpб6joint_model/prediction_output_1/BiasAdd/ReadVariableOpб5joint_model/prediction_output_1/MatMul/ReadVariableOpе
+joint_model/conv2d_45/Conv2D/ReadVariableOpReadVariableOp4joint_model_conv2d_45_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0К
joint_model/conv2d_45/Conv2DConv2Dinput_163joint_model/conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ъ
,joint_model/conv2d_45/BiasAdd/ReadVariableOpReadVariableOp5joint_model_conv2d_45_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┐
joint_model/conv2d_45/BiasAddBiasAdd%joint_model/conv2d_45/Conv2D:output:04joint_model/conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ё
joint_model/conv2d_45/ReluRelu&joint_model/conv2d_45/BiasAdd:output:0*
T0*/
_output_shapes
:         е
+joint_model/conv2d_42/Conv2D/ReadVariableOpReadVariableOp4joint_model_conv2d_42_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0К
joint_model/conv2d_42/Conv2DConv2Dinput_153joint_model/conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <*
paddingSAME*
strides
ъ
,joint_model/conv2d_42/BiasAdd/ReadVariableOpReadVariableOp5joint_model_conv2d_42_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┐
joint_model/conv2d_42/BiasAddBiasAdd%joint_model/conv2d_42/Conv2D:output:04joint_model/conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <ё
joint_model/conv2d_42/ReluRelu&joint_model/conv2d_42/BiasAdd:output:0*
T0*/
_output_shapes
:         <ў
(joint_model/spatial_dropout2d_7/IdentityIdentity(joint_model/conv2d_45/Relu:activations:0*
T0*/
_output_shapes
:         д
0joint_model/batch_normalization_7/ReadVariableOpReadVariableOp9joint_model_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype0ф
2joint_model/batch_normalization_7/ReadVariableOp_1ReadVariableOp;joint_model_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
Ajoint_model/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpJjoint_model_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╠
Cjoint_model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLjoint_model_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ѓ
2joint_model/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3(joint_model/conv2d_42/Relu:activations:08joint_model/batch_normalization_7/ReadVariableOp:value:0:joint_model/batch_normalization_7/ReadVariableOp_1:value:0Ijoint_model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Kjoint_model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         <:::::*
epsilon%oЃ:*
is_training( е
+joint_model/conv2d_46/Conv2D/ReadVariableOpReadVariableOp4joint_model_conv2d_46_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0­
joint_model/conv2d_46/Conv2DConv2D1joint_model/spatial_dropout2d_7/Identity:output:03joint_model/conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ъ
,joint_model/conv2d_46/BiasAdd/ReadVariableOpReadVariableOp5joint_model_conv2d_46_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┐
joint_model/conv2d_46/BiasAddBiasAdd%joint_model/conv2d_46/Conv2D:output:04joint_model/conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ё
joint_model/conv2d_46/ReluRelu&joint_model/conv2d_46/BiasAdd:output:0*
T0*/
_output_shapes
:         е
+joint_model/conv2d_43/Conv2D/ReadVariableOpReadVariableOp4joint_model_conv2d_43_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0ш
joint_model/conv2d_43/Conv2DConv2D6joint_model/batch_normalization_7/FusedBatchNormV3:y:03joint_model/conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <*
paddingSAME*
strides
ъ
,joint_model/conv2d_43/BiasAdd/ReadVariableOpReadVariableOp5joint_model_conv2d_43_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┐
joint_model/conv2d_43/BiasAddBiasAdd%joint_model/conv2d_43/Conv2D:output:04joint_model/conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <ё
joint_model/conv2d_43/ReluRelu&joint_model/conv2d_43/BiasAdd:output:0*
T0*/
_output_shapes
:         <е
+joint_model/conv2d_47/Conv2D/ReadVariableOpReadVariableOp4joint_model_conv2d_47_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0у
joint_model/conv2d_47/Conv2DConv2D(joint_model/conv2d_46/Relu:activations:03joint_model/conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ъ
,joint_model/conv2d_47/BiasAdd/ReadVariableOpReadVariableOp5joint_model_conv2d_47_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┐
joint_model/conv2d_47/BiasAddBiasAdd%joint_model/conv2d_47/Conv2D:output:04joint_model/conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ё
joint_model/conv2d_47/ReluRelu&joint_model/conv2d_47/BiasAdd:output:0*
T0*/
_output_shapes
:         е
+joint_model/conv2d_44/Conv2D/ReadVariableOpReadVariableOp4joint_model_conv2d_44_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0у
joint_model/conv2d_44/Conv2DConv2D(joint_model/conv2d_43/Relu:activations:03joint_model/conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <*
paddingSAME*
strides
ъ
,joint_model/conv2d_44/BiasAdd/ReadVariableOpReadVariableOp5joint_model_conv2d_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┐
joint_model/conv2d_44/BiasAddBiasAdd%joint_model/conv2d_44/Conv2D:output:04joint_model/conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <ё
joint_model/conv2d_44/ReluRelu&joint_model/conv2d_44/BiasAdd:output:0*
T0*/
_output_shapes
:         <Ѕ
8joint_model/global_max_pooling2d_7/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ╠
&joint_model/global_max_pooling2d_7/MaxMax(joint_model/conv2d_47/Relu:activations:0Ajoint_model/global_max_pooling2d_7/Max/reduction_indices:output:0*
T0*'
_output_shapes
:         m
joint_model/flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"       │
joint_model/flatten_15/ReshapeReshape/joint_model/global_max_pooling2d_7/Max:output:0%joint_model/flatten_15/Const:output:0*
T0*'
_output_shapes
:         m
joint_model/flatten_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"    x   г
joint_model/flatten_14/ReshapeReshape(joint_model/conv2d_44/Relu:activations:0%joint_model/flatten_14/Const:output:0*
T0*'
_output_shapes
:         xg
%joint_model/concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ж
 joint_model/concatenate_7/concatConcatV2'joint_model/flatten_15/Reshape:output:0'joint_model/flatten_14/Reshape:output:0.joint_model/concatenate_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:         zъ
*joint_model/dense_21/MatMul/ReadVariableOpReadVariableOp3joint_model_dense_21_matmul_readvariableop_resource*
_output_shapes

:z*
dtype0Х
joint_model/dense_21/MatMulMatMul)joint_model/concatenate_7/concat:output:02joint_model/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ю
+joint_model/dense_21/BiasAdd/ReadVariableOpReadVariableOp4joint_model_dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0х
joint_model/dense_21/BiasAddBiasAdd%joint_model/dense_21/MatMul:product:03joint_model/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         z
joint_model/dense_21/ReluRelu%joint_model/dense_21/BiasAdd:output:0*
T0*'
_output_shapes
:         ъ
*joint_model/dense_22/MatMul/ReadVariableOpReadVariableOp3joint_model_dense_22_matmul_readvariableop_resource*
_output_shapes

:*
dtype0┤
joint_model/dense_22/MatMulMatMul'joint_model/dense_21/Relu:activations:02joint_model/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ю
+joint_model/dense_22/BiasAdd/ReadVariableOpReadVariableOp4joint_model_dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0х
joint_model/dense_22/BiasAddBiasAdd%joint_model/dense_22/MatMul:product:03joint_model/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         z
joint_model/dense_22/ReluRelu%joint_model/dense_22/BiasAdd:output:0*
T0*'
_output_shapes
:         ъ
*joint_model/dense_23/MatMul/ReadVariableOpReadVariableOp3joint_model_dense_23_matmul_readvariableop_resource*
_output_shapes

:*
dtype0┤
joint_model/dense_23/MatMulMatMul'joint_model/dense_22/Relu:activations:02joint_model/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ю
+joint_model/dense_23/BiasAdd/ReadVariableOpReadVariableOp4joint_model_dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0х
joint_model/dense_23/BiasAddBiasAdd%joint_model/dense_23/MatMul:product:03joint_model/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         z
joint_model/dense_23/ReluRelu%joint_model/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:         ┤
5joint_model/prediction_output_1/MatMul/ReadVariableOpReadVariableOp>joint_model_prediction_output_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0╩
&joint_model/prediction_output_1/MatMulMatMul'joint_model/dense_23/Relu:activations:0=joint_model/prediction_output_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ▓
6joint_model/prediction_output_1/BiasAdd/ReadVariableOpReadVariableOp?joint_model_prediction_output_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0о
'joint_model/prediction_output_1/BiasAddBiasAdd0joint_model/prediction_output_1/MatMul:product:0>joint_model/prediction_output_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ┤
5joint_model/prediction_output_0/MatMul/ReadVariableOpReadVariableOp>joint_model_prediction_output_0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0╩
&joint_model/prediction_output_0/MatMulMatMul'joint_model/dense_23/Relu:activations:0=joint_model/prediction_output_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ▓
6joint_model/prediction_output_0/BiasAdd/ReadVariableOpReadVariableOp?joint_model_prediction_output_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0о
'joint_model/prediction_output_0/BiasAddBiasAdd0joint_model/prediction_output_0/MatMul:product:0>joint_model/prediction_output_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
IdentityIdentity0joint_model/prediction_output_0/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         Ђ

Identity_1Identity0joint_model/prediction_output_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ┘

NoOpNoOpB^joint_model/batch_normalization_7/FusedBatchNormV3/ReadVariableOpD^joint_model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_11^joint_model/batch_normalization_7/ReadVariableOp3^joint_model/batch_normalization_7/ReadVariableOp_1-^joint_model/conv2d_42/BiasAdd/ReadVariableOp,^joint_model/conv2d_42/Conv2D/ReadVariableOp-^joint_model/conv2d_43/BiasAdd/ReadVariableOp,^joint_model/conv2d_43/Conv2D/ReadVariableOp-^joint_model/conv2d_44/BiasAdd/ReadVariableOp,^joint_model/conv2d_44/Conv2D/ReadVariableOp-^joint_model/conv2d_45/BiasAdd/ReadVariableOp,^joint_model/conv2d_45/Conv2D/ReadVariableOp-^joint_model/conv2d_46/BiasAdd/ReadVariableOp,^joint_model/conv2d_46/Conv2D/ReadVariableOp-^joint_model/conv2d_47/BiasAdd/ReadVariableOp,^joint_model/conv2d_47/Conv2D/ReadVariableOp,^joint_model/dense_21/BiasAdd/ReadVariableOp+^joint_model/dense_21/MatMul/ReadVariableOp,^joint_model/dense_22/BiasAdd/ReadVariableOp+^joint_model/dense_22/MatMul/ReadVariableOp,^joint_model/dense_23/BiasAdd/ReadVariableOp+^joint_model/dense_23/MatMul/ReadVariableOp7^joint_model/prediction_output_0/BiasAdd/ReadVariableOp6^joint_model/prediction_output_0/MatMul/ReadVariableOp7^joint_model/prediction_output_1/BiasAdd/ReadVariableOp6^joint_model/prediction_output_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:         <:         <<: : : : : : : : : : : : : : : : : : : : : : : : : : 2є
Ajoint_model/batch_normalization_7/FusedBatchNormV3/ReadVariableOpAjoint_model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2і
Cjoint_model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Cjoint_model/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12d
0joint_model/batch_normalization_7/ReadVariableOp0joint_model/batch_normalization_7/ReadVariableOp2h
2joint_model/batch_normalization_7/ReadVariableOp_12joint_model/batch_normalization_7/ReadVariableOp_12\
,joint_model/conv2d_42/BiasAdd/ReadVariableOp,joint_model/conv2d_42/BiasAdd/ReadVariableOp2Z
+joint_model/conv2d_42/Conv2D/ReadVariableOp+joint_model/conv2d_42/Conv2D/ReadVariableOp2\
,joint_model/conv2d_43/BiasAdd/ReadVariableOp,joint_model/conv2d_43/BiasAdd/ReadVariableOp2Z
+joint_model/conv2d_43/Conv2D/ReadVariableOp+joint_model/conv2d_43/Conv2D/ReadVariableOp2\
,joint_model/conv2d_44/BiasAdd/ReadVariableOp,joint_model/conv2d_44/BiasAdd/ReadVariableOp2Z
+joint_model/conv2d_44/Conv2D/ReadVariableOp+joint_model/conv2d_44/Conv2D/ReadVariableOp2\
,joint_model/conv2d_45/BiasAdd/ReadVariableOp,joint_model/conv2d_45/BiasAdd/ReadVariableOp2Z
+joint_model/conv2d_45/Conv2D/ReadVariableOp+joint_model/conv2d_45/Conv2D/ReadVariableOp2\
,joint_model/conv2d_46/BiasAdd/ReadVariableOp,joint_model/conv2d_46/BiasAdd/ReadVariableOp2Z
+joint_model/conv2d_46/Conv2D/ReadVariableOp+joint_model/conv2d_46/Conv2D/ReadVariableOp2\
,joint_model/conv2d_47/BiasAdd/ReadVariableOp,joint_model/conv2d_47/BiasAdd/ReadVariableOp2Z
+joint_model/conv2d_47/Conv2D/ReadVariableOp+joint_model/conv2d_47/Conv2D/ReadVariableOp2Z
+joint_model/dense_21/BiasAdd/ReadVariableOp+joint_model/dense_21/BiasAdd/ReadVariableOp2X
*joint_model/dense_21/MatMul/ReadVariableOp*joint_model/dense_21/MatMul/ReadVariableOp2Z
+joint_model/dense_22/BiasAdd/ReadVariableOp+joint_model/dense_22/BiasAdd/ReadVariableOp2X
*joint_model/dense_22/MatMul/ReadVariableOp*joint_model/dense_22/MatMul/ReadVariableOp2Z
+joint_model/dense_23/BiasAdd/ReadVariableOp+joint_model/dense_23/BiasAdd/ReadVariableOp2X
*joint_model/dense_23/MatMul/ReadVariableOp*joint_model/dense_23/MatMul/ReadVariableOp2p
6joint_model/prediction_output_0/BiasAdd/ReadVariableOp6joint_model/prediction_output_0/BiasAdd/ReadVariableOp2n
5joint_model/prediction_output_0/MatMul/ReadVariableOp5joint_model/prediction_output_0/MatMul/ReadVariableOp2p
6joint_model/prediction_output_1/BiasAdd/ReadVariableOp6joint_model/prediction_output_1/BiasAdd/ReadVariableOp2n
5joint_model/prediction_output_1/MatMul/ReadVariableOp5joint_model/prediction_output_1/MatMul/ReadVariableOp:Y U
/
_output_shapes
:         <
"
_user_specified_name
input_15:YU
/
_output_shapes
:         <<
"
_user_specified_name
input_16
ІT
┐
G__inference_joint_model_layer_call_and_return_conditional_losses_716690

inputs
inputs_1*
conv2d_45_716479:		
conv2d_45_716481:*
conv2d_42_716496:
conv2d_42_716498:*
batch_normalization_7_716502:*
batch_normalization_7_716504:*
batch_normalization_7_716506:*
batch_normalization_7_716508:*
conv2d_46_716523:
conv2d_46_716525:*
conv2d_43_716540:	
conv2d_43_716542:*
conv2d_47_716557:
conv2d_47_716559:*
conv2d_44_716574:
conv2d_44_716576:!
dense_21_716617:z
dense_21_716619:!
dense_22_716634:
dense_22_716636:!
dense_23_716651:
dense_23_716653:,
prediction_output_1_716667:(
prediction_output_1_716669:,
prediction_output_0_716683:(
prediction_output_0_716685:
identity

identity_1ѕб-batch_normalization_7/StatefulPartitionedCallб!conv2d_42/StatefulPartitionedCallб!conv2d_43/StatefulPartitionedCallб!conv2d_44/StatefulPartitionedCallб!conv2d_45/StatefulPartitionedCallб!conv2d_46/StatefulPartitionedCallб!conv2d_47/StatefulPartitionedCallб dense_21/StatefulPartitionedCallб dense_22/StatefulPartitionedCallб dense_23/StatefulPartitionedCallб+prediction_output_0/StatefulPartitionedCallб+prediction_output_1/StatefulPartitionedCallЂ
!conv2d_45/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv2d_45_716479conv2d_45_716481*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_45_layer_call_and_return_conditional_losses_716478 
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_42_716496conv2d_42_716498*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         <*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_42_layer_call_and_return_conditional_losses_716495§
#spatial_dropout2d_7/PartitionedCallPartitionedCall*conv2d_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_spatial_dropout2d_7_layer_call_and_return_conditional_losses_716350Њ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0batch_normalization_7_716502batch_normalization_7_716504batch_normalization_7_716506batch_normalization_7_716508*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         <*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_716403Ц
!conv2d_46/StatefulPartitionedCallStatefulPartitionedCall,spatial_dropout2d_7/PartitionedCall:output:0conv2d_46_716523conv2d_46_716525*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_46_layer_call_and_return_conditional_losses_716522»
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv2d_43_716540conv2d_43_716542*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         <*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_43_layer_call_and_return_conditional_losses_716539Б
!conv2d_47/StatefulPartitionedCallStatefulPartitionedCall*conv2d_46/StatefulPartitionedCall:output:0conv2d_47_716557conv2d_47_716559*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_47_layer_call_and_return_conditional_losses_716556Б
!conv2d_44/StatefulPartitionedCallStatefulPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0conv2d_44_716574conv2d_44_716576*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         <*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_44_layer_call_and_return_conditional_losses_716573ч
&global_max_pooling2d_7/PartitionedCallPartitionedCall*conv2d_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *[
fVRT
R__inference_global_max_pooling2d_7_layer_call_and_return_conditional_losses_716455У
flatten_15/PartitionedCallPartitionedCall/global_max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_flatten_15_layer_call_and_return_conditional_losses_716586с
flatten_14/PartitionedCallPartitionedCall*conv2d_44/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         x* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_flatten_14_layer_call_and_return_conditional_losses_716594ѕ
concatenate_7/PartitionedCallPartitionedCall#flatten_15/PartitionedCall:output:0#flatten_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         z* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_concatenate_7_layer_call_and_return_conditional_losses_716603Њ
 dense_21/StatefulPartitionedCallStatefulPartitionedCall&concatenate_7/PartitionedCall:output:0dense_21_716617dense_21_716619*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_716616ќ
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_716634dense_22_716636*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_716633ќ
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_716651dense_23_716653*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_716650┬
+prediction_output_1/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0prediction_output_1_716667prediction_output_1_716669*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_prediction_output_1_layer_call_and_return_conditional_losses_716666┬
+prediction_output_0/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0prediction_output_0_716683prediction_output_0_716685*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *X
fSRQ
O__inference_prediction_output_0_layer_call_and_return_conditional_losses_716682Ѓ
IdentityIdentity4prediction_output_0/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Ё

Identity_1Identity4prediction_output_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Њ
NoOpNoOp.^batch_normalization_7/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall"^conv2d_44/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall,^prediction_output_0/StatefulPartitionedCall,^prediction_output_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:         <:         <<: : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2F
!conv2d_44/StatefulPartitionedCall!conv2d_44/StatefulPartitionedCall2F
!conv2d_45/StatefulPartitionedCall!conv2d_45/StatefulPartitionedCall2F
!conv2d_46/StatefulPartitionedCall!conv2d_46/StatefulPartitionedCall2F
!conv2d_47/StatefulPartitionedCall!conv2d_47/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2Z
+prediction_output_0/StatefulPartitionedCall+prediction_output_0/StatefulPartitionedCall2Z
+prediction_output_1/StatefulPartitionedCall+prediction_output_1/StatefulPartitionedCall:W S
/
_output_shapes
:         <
 
_user_specified_nameinputs:WS
/
_output_shapes
:         <<
 
_user_specified_nameinputs
┼
ќ
)__inference_dense_22_layer_call_fn_717990

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_716633o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Рц
╗$
__inference__traced_save_718339
file_prefix/
+savev2_conv2d_45_kernel_read_readvariableop-
)savev2_conv2d_45_bias_read_readvariableop/
+savev2_conv2d_42_kernel_read_readvariableop-
)savev2_conv2d_42_bias_read_readvariableop/
+savev2_conv2d_46_kernel_read_readvariableop-
)savev2_conv2d_46_bias_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop/
+savev2_conv2d_47_kernel_read_readvariableop-
)savev2_conv2d_47_bias_read_readvariableop/
+savev2_conv2d_43_kernel_read_readvariableop-
)savev2_conv2d_43_bias_read_readvariableop/
+savev2_conv2d_44_kernel_read_readvariableop-
)savev2_conv2d_44_bias_read_readvariableop.
*savev2_dense_21_kernel_read_readvariableop,
(savev2_dense_21_bias_read_readvariableop.
*savev2_dense_22_kernel_read_readvariableop,
(savev2_dense_22_bias_read_readvariableop.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableop9
5savev2_prediction_output_0_kernel_read_readvariableop7
3savev2_prediction_output_0_bias_read_readvariableop9
5savev2_prediction_output_1_kernel_read_readvariableop7
3savev2_prediction_output_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv2d_45_kernel_m_read_readvariableop4
0savev2_adam_conv2d_45_bias_m_read_readvariableop6
2savev2_adam_conv2d_42_kernel_m_read_readvariableop4
0savev2_adam_conv2d_42_bias_m_read_readvariableop6
2savev2_adam_conv2d_46_kernel_m_read_readvariableop4
0savev2_adam_conv2d_46_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_m_read_readvariableop6
2savev2_adam_conv2d_47_kernel_m_read_readvariableop4
0savev2_adam_conv2d_47_bias_m_read_readvariableop6
2savev2_adam_conv2d_43_kernel_m_read_readvariableop4
0savev2_adam_conv2d_43_bias_m_read_readvariableop6
2savev2_adam_conv2d_44_kernel_m_read_readvariableop4
0savev2_adam_conv2d_44_bias_m_read_readvariableop5
1savev2_adam_dense_21_kernel_m_read_readvariableop3
/savev2_adam_dense_21_bias_m_read_readvariableop5
1savev2_adam_dense_22_kernel_m_read_readvariableop3
/savev2_adam_dense_22_bias_m_read_readvariableop5
1savev2_adam_dense_23_kernel_m_read_readvariableop3
/savev2_adam_dense_23_bias_m_read_readvariableop@
<savev2_adam_prediction_output_0_kernel_m_read_readvariableop>
:savev2_adam_prediction_output_0_bias_m_read_readvariableop@
<savev2_adam_prediction_output_1_kernel_m_read_readvariableop>
:savev2_adam_prediction_output_1_bias_m_read_readvariableop6
2savev2_adam_conv2d_45_kernel_v_read_readvariableop4
0savev2_adam_conv2d_45_bias_v_read_readvariableop6
2savev2_adam_conv2d_42_kernel_v_read_readvariableop4
0savev2_adam_conv2d_42_bias_v_read_readvariableop6
2savev2_adam_conv2d_46_kernel_v_read_readvariableop4
0savev2_adam_conv2d_46_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_v_read_readvariableop6
2savev2_adam_conv2d_47_kernel_v_read_readvariableop4
0savev2_adam_conv2d_47_bias_v_read_readvariableop6
2savev2_adam_conv2d_43_kernel_v_read_readvariableop4
0savev2_adam_conv2d_43_bias_v_read_readvariableop6
2savev2_adam_conv2d_44_kernel_v_read_readvariableop4
0savev2_adam_conv2d_44_bias_v_read_readvariableop5
1savev2_adam_dense_21_kernel_v_read_readvariableop3
/savev2_adam_dense_21_bias_v_read_readvariableop5
1savev2_adam_dense_22_kernel_v_read_readvariableop3
/savev2_adam_dense_22_bias_v_read_readvariableop5
1savev2_adam_dense_23_kernel_v_read_readvariableop3
/savev2_adam_dense_23_bias_v_read_readvariableop@
<savev2_adam_prediction_output_0_kernel_v_read_readvariableop>
:savev2_adam_prediction_output_0_bias_v_read_readvariableop@
<savev2_adam_prediction_output_1_kernel_v_read_readvariableop>
:savev2_adam_prediction_output_1_bias_v_read_readvariableop
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
: Ы/
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:V*
dtype0*Џ/
valueЉ/Bј/VB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHю
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:V*
dtype0*┴
valueиB┤VB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ђ#
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_45_kernel_read_readvariableop)savev2_conv2d_45_bias_read_readvariableop+savev2_conv2d_42_kernel_read_readvariableop)savev2_conv2d_42_bias_read_readvariableop+savev2_conv2d_46_kernel_read_readvariableop)savev2_conv2d_46_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop+savev2_conv2d_47_kernel_read_readvariableop)savev2_conv2d_47_bias_read_readvariableop+savev2_conv2d_43_kernel_read_readvariableop)savev2_conv2d_43_bias_read_readvariableop+savev2_conv2d_44_kernel_read_readvariableop)savev2_conv2d_44_bias_read_readvariableop*savev2_dense_21_kernel_read_readvariableop(savev2_dense_21_bias_read_readvariableop*savev2_dense_22_kernel_read_readvariableop(savev2_dense_22_bias_read_readvariableop*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop5savev2_prediction_output_0_kernel_read_readvariableop3savev2_prediction_output_0_bias_read_readvariableop5savev2_prediction_output_1_kernel_read_readvariableop3savev2_prediction_output_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv2d_45_kernel_m_read_readvariableop0savev2_adam_conv2d_45_bias_m_read_readvariableop2savev2_adam_conv2d_42_kernel_m_read_readvariableop0savev2_adam_conv2d_42_bias_m_read_readvariableop2savev2_adam_conv2d_46_kernel_m_read_readvariableop0savev2_adam_conv2d_46_bias_m_read_readvariableop=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop<savev2_adam_batch_normalization_7_beta_m_read_readvariableop2savev2_adam_conv2d_47_kernel_m_read_readvariableop0savev2_adam_conv2d_47_bias_m_read_readvariableop2savev2_adam_conv2d_43_kernel_m_read_readvariableop0savev2_adam_conv2d_43_bias_m_read_readvariableop2savev2_adam_conv2d_44_kernel_m_read_readvariableop0savev2_adam_conv2d_44_bias_m_read_readvariableop1savev2_adam_dense_21_kernel_m_read_readvariableop/savev2_adam_dense_21_bias_m_read_readvariableop1savev2_adam_dense_22_kernel_m_read_readvariableop/savev2_adam_dense_22_bias_m_read_readvariableop1savev2_adam_dense_23_kernel_m_read_readvariableop/savev2_adam_dense_23_bias_m_read_readvariableop<savev2_adam_prediction_output_0_kernel_m_read_readvariableop:savev2_adam_prediction_output_0_bias_m_read_readvariableop<savev2_adam_prediction_output_1_kernel_m_read_readvariableop:savev2_adam_prediction_output_1_bias_m_read_readvariableop2savev2_adam_conv2d_45_kernel_v_read_readvariableop0savev2_adam_conv2d_45_bias_v_read_readvariableop2savev2_adam_conv2d_42_kernel_v_read_readvariableop0savev2_adam_conv2d_42_bias_v_read_readvariableop2savev2_adam_conv2d_46_kernel_v_read_readvariableop0savev2_adam_conv2d_46_bias_v_read_readvariableop=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop<savev2_adam_batch_normalization_7_beta_v_read_readvariableop2savev2_adam_conv2d_47_kernel_v_read_readvariableop0savev2_adam_conv2d_47_bias_v_read_readvariableop2savev2_adam_conv2d_43_kernel_v_read_readvariableop0savev2_adam_conv2d_43_bias_v_read_readvariableop2savev2_adam_conv2d_44_kernel_v_read_readvariableop0savev2_adam_conv2d_44_bias_v_read_readvariableop1savev2_adam_dense_21_kernel_v_read_readvariableop/savev2_adam_dense_21_bias_v_read_readvariableop1savev2_adam_dense_22_kernel_v_read_readvariableop/savev2_adam_dense_22_bias_v_read_readvariableop1savev2_adam_dense_23_kernel_v_read_readvariableop/savev2_adam_dense_23_bias_v_read_readvariableop<savev2_adam_prediction_output_0_kernel_v_read_readvariableop:savev2_adam_prediction_output_0_bias_v_read_readvariableop<savev2_adam_prediction_output_1_kernel_v_read_readvariableop:savev2_adam_prediction_output_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *d
dtypesZ
X2V	љ
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

identity_1Identity_1:output:0* 
_input_shapesь
Ж: :		::::::::::::	::::z:::::::::: : : : : : : : : : : :		::::::::::	::::z::::::::::		::::::::::	::::z:::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:		: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:	: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:z: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :,&(
&
_output_shapes
:		: '

_output_shapes
::,((
&
_output_shapes
:: )

_output_shapes
::,*(
&
_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
::,.(
&
_output_shapes
:: /

_output_shapes
::,0(
&
_output_shapes
:	: 1

_output_shapes
::,2(
&
_output_shapes
:: 3

_output_shapes
::$4 

_output_shapes

:z: 5

_output_shapes
::$6 

_output_shapes

:: 7

_output_shapes
::$8 

_output_shapes

:: 9

_output_shapes
::$: 

_output_shapes

:: ;

_output_shapes
::$< 

_output_shapes

:: =

_output_shapes
::,>(
&
_output_shapes
:		: ?

_output_shapes
::,@(
&
_output_shapes
:: A

_output_shapes
::,B(
&
_output_shapes
:: C

_output_shapes
:: D

_output_shapes
:: E

_output_shapes
::,F(
&
_output_shapes
:: G

_output_shapes
::,H(
&
_output_shapes
:	: I

_output_shapes
::,J(
&
_output_shapes
:: K

_output_shapes
::$L 

_output_shapes

:z: M

_output_shapes
::$N 

_output_shapes

:: O

_output_shapes
::$P 

_output_shapes

:: Q

_output_shapes
::$R 

_output_shapes

:: S

_output_shapes
::$T 

_output_shapes

:: U

_output_shapes
::V

_output_shapes
: 
м	
ђ
O__inference_prediction_output_1_layer_call_and_return_conditional_losses_718059

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
є
└
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_717855

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
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
-:+                           н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
я
Э
,__inference_joint_model_layer_call_fn_717469
inputs_0
inputs_1!
unknown:		
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:	

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:z

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:
identity

identity_1ѕбStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         :         *:
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_joint_model_layer_call_and_return_conditional_losses_717014o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j:         <:         <<: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:         <
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:         <<
"
_user_specified_name
inputs/1
┬
u
I__inference_concatenate_7_layer_call_and_return_conditional_losses_717961
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:         zW
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:         z"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         :         x:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         x
"
_user_specified_name
inputs/1
■
n
O__inference_spatial_dropout2d_7_layer_call_and_return_conditional_losses_716378

inputs
identityѕ;
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
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
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
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?Є
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4                                    `
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :`
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :о
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:г
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"                  *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>и
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"                  ђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"                  ї
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4                                    |
IdentityIdentitydropout/Mul_1:z:0*
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
№
Ъ
*__inference_conv2d_46_layer_call_fn_717782

inputs!
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_46_layer_call_and_return_conditional_losses_716522w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ё
■
E__inference_conv2d_42_layer_call_and_return_conditional_losses_717773

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         <i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         <w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         <: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         <
 
_user_specified_nameinputs
ё
■
E__inference_conv2d_47_layer_call_and_return_conditional_losses_717875

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
њ	
Л
6__inference_batch_normalization_7_layer_call_fn_717819

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_716434Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Џ

ш
D__inference_dense_21_layer_call_and_return_conditional_losses_717981

inputs0
matmul_readvariableop_resource:z-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:z*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         z
 
_user_specified_nameinputs"х	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*л
serving_default╝
E
input_159
serving_default_input_15:0         <
E
input_169
serving_default_input_16:0         <<G
prediction_output_00
StatefulPartitionedCall:0         G
prediction_output_10
StatefulPartitionedCall:1         tensorflow/serving/predict:Љб
Ю
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer-11
layer-12
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
layer_with_weights-10
layer-17
layer_with_weights-11
layer-18
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
П
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias
 %_jit_compiled_convolution_op"
_tf_keras_layer
"
_tf_keras_input_layer
╝
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,_random_generator"
_tf_keras_layer
П
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias
 5_jit_compiled_convolution_op"
_tf_keras_layer
П
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

<kernel
=bias
 >_jit_compiled_convolution_op"
_tf_keras_layer
Ж
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses
Eaxis
	Fgamma
Gbeta
Hmoving_mean
Imoving_variance"
_tf_keras_layer
П
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses

Pkernel
Qbias
 R_jit_compiled_convolution_op"
_tf_keras_layer
П
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias
 [_jit_compiled_convolution_op"
_tf_keras_layer
Ц
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses"
_tf_keras_layer
П
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias
 j_jit_compiled_convolution_op"
_tf_keras_layer
Ц
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses"
_tf_keras_layer
Ц
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses"
_tf_keras_layer
Ц
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses"
_tf_keras_layer
└
}	variables
~trainable_variables
regularization_losses
ђ	keras_api
Ђ__call__
+ѓ&call_and_return_all_conditional_losses
Ѓkernel
	ёbias"
_tf_keras_layer
├
Ё	variables
єtrainable_variables
Єregularization_losses
ѕ	keras_api
Ѕ__call__
+і&call_and_return_all_conditional_losses
Іkernel
	їbias"
_tf_keras_layer
├
Ї	variables
јtrainable_variables
Јregularization_losses
љ	keras_api
Љ__call__
+њ&call_and_return_all_conditional_losses
Њkernel
	ћbias"
_tf_keras_layer
├
Ћ	variables
ќtrainable_variables
Ќregularization_losses
ў	keras_api
Ў__call__
+џ&call_and_return_all_conditional_losses
Џkernel
	юbias"
_tf_keras_layer
├
Ю	variables
ъtrainable_variables
Ъregularization_losses
а	keras_api
А__call__
+б&call_and_return_all_conditional_losses
Бkernel
	цbias"
_tf_keras_layer
­
#0
$1
32
43
<4
=5
F6
G7
H8
I9
P10
Q11
Y12
Z13
h14
i15
Ѓ16
ё17
І18
ї19
Њ20
ћ21
Џ22
ю23
Б24
ц25"
trackable_list_wrapper
Я
#0
$1
32
43
<4
=5
F6
G7
P8
Q9
Y10
Z11
h12
i13
Ѓ14
ё15
І16
ї17
Њ18
ћ19
Џ20
ю21
Б22
ц23"
trackable_list_wrapper
 "
trackable_list_wrapper
¤
Цnon_trainable_variables
дlayers
Дmetrics
 еlayer_regularization_losses
Еlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ь
фtrace_0
Фtrace_1
гtrace_2
Гtrace_32Щ
,__inference_joint_model_layer_call_fn_716747
,__inference_joint_model_layer_call_fn_717409
,__inference_joint_model_layer_call_fn_717469
,__inference_joint_model_layer_call_fn_717131┐
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
 zфtrace_0zФtrace_1zгtrace_2zГtrace_3
┘
«trace_0
»trace_1
░trace_2
▒trace_32Т
G__inference_joint_model_layer_call_and_return_conditional_losses_717573
G__inference_joint_model_layer_call_and_return_conditional_losses_717695
G__inference_joint_model_layer_call_and_return_conditional_losses_717206
G__inference_joint_model_layer_call_and_return_conditional_losses_717281┐
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
 z«trace_0z»trace_1z░trace_2z▒trace_3
ОBн
!__inference__wrapped_model_716341input_15input_16"ў
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
╠
	▓iter
│beta_1
┤beta_2

хdecay
Хlearning_rate#m┬$m├3m─4m┼<mк=mКFm╚Gm╔Pm╩Qm╦Ym╠Zm═hm╬im¤	Ѓmл	ёmЛ	Іmм	їmМ	Њmн	ћmН	Џmо	юmО	Бmп	цm┘#v┌$v█3v▄4vП<vя=v▀FvЯGvрPvРQvсYvСZvтhvТivу	ЃvУ	ёvж	ІvЖ	їvв	ЊvВ	ћvь	ЏvЬ	юv№	Бv­	цvы"
	optimizer
-
иserving_default"
signature_map
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
▓
Иnon_trainable_variables
╣layers
║metrics
 ╗layer_regularization_losses
╝layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
­
йtrace_02Л
*__inference_conv2d_45_layer_call_fn_717704б
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
 zйtrace_0
І
Йtrace_02В
E__inference_conv2d_45_layer_call_and_return_conditional_losses_717715б
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
 zЙtrace_0
*:(		2conv2d_45/kernel
:2conv2d_45/bias
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
▓
┐non_trainable_variables
└layers
┴metrics
 ┬layer_regularization_losses
├layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
П
─trace_0
┼trace_12б
4__inference_spatial_dropout2d_7_layer_call_fn_717720
4__inference_spatial_dropout2d_7_layer_call_fn_717725│
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
 z─trace_0z┼trace_1
Њ
кtrace_0
Кtrace_12п
O__inference_spatial_dropout2d_7_layer_call_and_return_conditional_losses_717730
O__inference_spatial_dropout2d_7_layer_call_and_return_conditional_losses_717753│
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
 zкtrace_0zКtrace_1
"
_generic_user_object
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╚non_trainable_variables
╔layers
╩metrics
 ╦layer_regularization_losses
╠layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
­
═trace_02Л
*__inference_conv2d_42_layer_call_fn_717762б
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
 z═trace_0
І
╬trace_02В
E__inference_conv2d_42_layer_call_and_return_conditional_losses_717773б
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
 z╬trace_0
*:(2conv2d_42/kernel
:2conv2d_42/bias
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
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
¤non_trainable_variables
лlayers
Лmetrics
 мlayer_regularization_losses
Мlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
­
нtrace_02Л
*__inference_conv2d_46_layer_call_fn_717782б
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
 zнtrace_0
І
Нtrace_02В
E__inference_conv2d_46_layer_call_and_return_conditional_losses_717793б
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
 zНtrace_0
*:(2conv2d_46/kernel
:2conv2d_46/bias
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
<
F0
G1
H2
I3"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
оnon_trainable_variables
Оlayers
пmetrics
 ┘layer_regularization_losses
┌layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
р
█trace_0
▄trace_12д
6__inference_batch_normalization_7_layer_call_fn_717806
6__inference_batch_normalization_7_layer_call_fn_717819│
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
 z█trace_0z▄trace_1
Ќ
Пtrace_0
яtrace_12▄
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_717837
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_717855│
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
 zПtrace_0zяtrace_1
 "
trackable_list_wrapper
):'2batch_normalization_7/gamma
(:&2batch_normalization_7/beta
1:/ (2!batch_normalization_7/moving_mean
5:3 (2%batch_normalization_7/moving_variance
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
▀non_trainable_variables
Яlayers
рmetrics
 Рlayer_regularization_losses
сlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
­
Сtrace_02Л
*__inference_conv2d_47_layer_call_fn_717864б
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
 zСtrace_0
І
тtrace_02В
E__inference_conv2d_47_layer_call_and_return_conditional_losses_717875б
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
 zтtrace_0
*:(2conv2d_47/kernel
:2conv2d_47/bias
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
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Тnon_trainable_variables
уlayers
Уmetrics
 жlayer_regularization_losses
Жlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
­
вtrace_02Л
*__inference_conv2d_43_layer_call_fn_717884б
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
 zвtrace_0
І
Вtrace_02В
E__inference_conv2d_43_layer_call_and_return_conditional_losses_717895б
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
 zВtrace_0
*:(	2conv2d_43/kernel
:2conv2d_43/bias
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
▓
ьnon_trainable_variables
Ьlayers
№metrics
 ­layer_regularization_losses
ыlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
§
Ыtrace_02я
7__inference_global_max_pooling2d_7_layer_call_fn_717900б
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
 zЫtrace_0
ў
зtrace_02щ
R__inference_global_max_pooling2d_7_layer_call_and_return_conditional_losses_717906б
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
 zзtrace_0
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Зnon_trainable_variables
шlayers
Шmetrics
 эlayer_regularization_losses
Эlayer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
­
щtrace_02Л
*__inference_conv2d_44_layer_call_fn_717915б
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
 zщtrace_0
І
Щtrace_02В
E__inference_conv2d_44_layer_call_and_return_conditional_losses_717926б
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
 zЩtrace_0
*:(2conv2d_44/kernel
:2conv2d_44/bias
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
▓
чnon_trainable_variables
Чlayers
§metrics
 ■layer_regularization_losses
 layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
ы
ђtrace_02м
+__inference_flatten_15_layer_call_fn_717931б
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
 zђtrace_0
ї
Ђtrace_02ь
F__inference_flatten_15_layer_call_and_return_conditional_losses_717937б
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ѓnon_trainable_variables
Ѓlayers
ёmetrics
 Ёlayer_regularization_losses
єlayer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
ы
Єtrace_02м
+__inference_flatten_14_layer_call_fn_717942б
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
 zЄtrace_0
ї
ѕtrace_02ь
F__inference_flatten_14_layer_call_and_return_conditional_losses_717948б
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
 zѕtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Ѕnon_trainable_variables
іlayers
Іmetrics
 їlayer_regularization_losses
Їlayer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
З
јtrace_02Н
.__inference_concatenate_7_layer_call_fn_717954б
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
 zјtrace_0
Ј
Јtrace_02­
I__inference_concatenate_7_layer_call_and_return_conditional_losses_717961б
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
 zЈtrace_0
0
Ѓ0
ё1"
trackable_list_wrapper
0
Ѓ0
ё1"
trackable_list_wrapper
 "
trackable_list_wrapper
х
љnon_trainable_variables
Љlayers
њmetrics
 Њlayer_regularization_losses
ћlayer_metrics
}	variables
~trainable_variables
regularization_losses
Ђ__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses"
_generic_user_object
№
Ћtrace_02л
)__inference_dense_21_layer_call_fn_717970б
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
 zЋtrace_0
і
ќtrace_02в
D__inference_dense_21_layer_call_and_return_conditional_losses_717981б
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
 zќtrace_0
!:z2dense_21/kernel
:2dense_21/bias
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
Ќnon_trainable_variables
ўlayers
Ўmetrics
 џlayer_regularization_losses
Џlayer_metrics
Ё	variables
єtrainable_variables
Єregularization_losses
Ѕ__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses"
_generic_user_object
№
юtrace_02л
)__inference_dense_22_layer_call_fn_717990б
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
 zюtrace_0
і
Юtrace_02в
D__inference_dense_22_layer_call_and_return_conditional_losses_718001б
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
 zЮtrace_0
!:2dense_22/kernel
:2dense_22/bias
0
Њ0
ћ1"
trackable_list_wrapper
0
Њ0
ћ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ъnon_trainable_variables
Ъlayers
аmetrics
 Аlayer_regularization_losses
бlayer_metrics
Ї	variables
јtrainable_variables
Јregularization_losses
Љ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses"
_generic_user_object
№
Бtrace_02л
)__inference_dense_23_layer_call_fn_718010б
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
 zБtrace_0
і
цtrace_02в
D__inference_dense_23_layer_call_and_return_conditional_losses_718021б
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
 zцtrace_0
!:2dense_23/kernel
:2dense_23/bias
0
Џ0
ю1"
trackable_list_wrapper
0
Џ0
ю1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Цnon_trainable_variables
дlayers
Дmetrics
 еlayer_regularization_losses
Еlayer_metrics
Ћ	variables
ќtrainable_variables
Ќregularization_losses
Ў__call__
+џ&call_and_return_all_conditional_losses
'џ"call_and_return_conditional_losses"
_generic_user_object
Щ
фtrace_02█
4__inference_prediction_output_0_layer_call_fn_718030б
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
 zфtrace_0
Ћ
Фtrace_02Ш
O__inference_prediction_output_0_layer_call_and_return_conditional_losses_718040б
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
 zФtrace_0
,:*2prediction_output_0/kernel
&:$2prediction_output_0/bias
0
Б0
ц1"
trackable_list_wrapper
0
Б0
ц1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
гnon_trainable_variables
Гlayers
«metrics
 »layer_regularization_losses
░layer_metrics
Ю	variables
ъtrainable_variables
Ъregularization_losses
А__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
Щ
▒trace_02█
4__inference_prediction_output_1_layer_call_fn_718049б
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
 z▒trace_0
Ћ
▓trace_02Ш
O__inference_prediction_output_1_layer_call_and_return_conditional_losses_718059б
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
 z▓trace_0
,:*2prediction_output_1/kernel
&:$2prediction_output_1/bias
.
H0
I1"
trackable_list_wrapper
«
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
18"
trackable_list_wrapper
8
│0
┤1
х2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЅBє
,__inference_joint_model_layer_call_fn_716747input_15input_16"┐
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
ЅBє
,__inference_joint_model_layer_call_fn_717409inputs/0inputs/1"┐
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
ЅBє
,__inference_joint_model_layer_call_fn_717469inputs/0inputs/1"┐
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
ЅBє
,__inference_joint_model_layer_call_fn_717131input_15input_16"┐
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
цBА
G__inference_joint_model_layer_call_and_return_conditional_losses_717573inputs/0inputs/1"┐
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
цBА
G__inference_joint_model_layer_call_and_return_conditional_losses_717695inputs/0inputs/1"┐
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
цBА
G__inference_joint_model_layer_call_and_return_conditional_losses_717206input_15input_16"┐
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
цBА
G__inference_joint_model_layer_call_and_return_conditional_losses_717281input_15input_16"┐
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
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
нBЛ
$__inference_signature_wrapper_717349input_15input_16"ћ
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
*__inference_conv2d_45_layer_call_fn_717704inputs"б
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
E__inference_conv2d_45_layer_call_and_return_conditional_losses_717715inputs"б
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
щBШ
4__inference_spatial_dropout2d_7_layer_call_fn_717720inputs"│
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
щBШ
4__inference_spatial_dropout2d_7_layer_call_fn_717725inputs"│
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
ћBЉ
O__inference_spatial_dropout2d_7_layer_call_and_return_conditional_losses_717730inputs"│
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
ћBЉ
O__inference_spatial_dropout2d_7_layer_call_and_return_conditional_losses_717753inputs"│
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
яB█
*__inference_conv2d_42_layer_call_fn_717762inputs"б
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
E__inference_conv2d_42_layer_call_and_return_conditional_losses_717773inputs"б
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
*__inference_conv2d_46_layer_call_fn_717782inputs"б
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
E__inference_conv2d_46_layer_call_and_return_conditional_losses_717793inputs"б
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
.
H0
I1"
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
6__inference_batch_normalization_7_layer_call_fn_717806inputs"│
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
6__inference_batch_normalization_7_layer_call_fn_717819inputs"│
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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_717837inputs"│
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
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_717855inputs"│
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
яB█
*__inference_conv2d_47_layer_call_fn_717864inputs"б
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
E__inference_conv2d_47_layer_call_and_return_conditional_losses_717875inputs"б
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
*__inference_conv2d_43_layer_call_fn_717884inputs"б
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
E__inference_conv2d_43_layer_call_and_return_conditional_losses_717895inputs"б
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
вBУ
7__inference_global_max_pooling2d_7_layer_call_fn_717900inputs"б
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
єBЃ
R__inference_global_max_pooling2d_7_layer_call_and_return_conditional_losses_717906inputs"б
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
*__inference_conv2d_44_layer_call_fn_717915inputs"б
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
E__inference_conv2d_44_layer_call_and_return_conditional_losses_717926inputs"б
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
▀B▄
+__inference_flatten_15_layer_call_fn_717931inputs"б
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
F__inference_flatten_15_layer_call_and_return_conditional_losses_717937inputs"б
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
▀B▄
+__inference_flatten_14_layer_call_fn_717942inputs"б
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
F__inference_flatten_14_layer_call_and_return_conditional_losses_717948inputs"б
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
ЬBв
.__inference_concatenate_7_layer_call_fn_717954inputs/0inputs/1"б
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
ЅBє
I__inference_concatenate_7_layer_call_and_return_conditional_losses_717961inputs/0inputs/1"б
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
)__inference_dense_21_layer_call_fn_717970inputs"б
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
D__inference_dense_21_layer_call_and_return_conditional_losses_717981inputs"б
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
)__inference_dense_22_layer_call_fn_717990inputs"б
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
D__inference_dense_22_layer_call_and_return_conditional_losses_718001inputs"б
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
)__inference_dense_23_layer_call_fn_718010inputs"б
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
D__inference_dense_23_layer_call_and_return_conditional_losses_718021inputs"б
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
УBт
4__inference_prediction_output_0_layer_call_fn_718030inputs"б
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
ЃBђ
O__inference_prediction_output_0_layer_call_and_return_conditional_losses_718040inputs"б
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
УBт
4__inference_prediction_output_1_layer_call_fn_718049inputs"б
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
ЃBђ
O__inference_prediction_output_1_layer_call_and_return_conditional_losses_718059inputs"б
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
R
Х	variables
и	keras_api

Иtotal

╣count"
_tf_keras_metric
R
║	variables
╗	keras_api

╝total

йcount"
_tf_keras_metric
R
Й	variables
┐	keras_api

└total

┴count"
_tf_keras_metric
0
И0
╣1"
trackable_list_wrapper
.
Х	variables"
_generic_user_object
:  (2total
:  (2count
0
╝0
й1"
trackable_list_wrapper
.
║	variables"
_generic_user_object
:  (2total
:  (2count
0
└0
┴1"
trackable_list_wrapper
.
Й	variables"
_generic_user_object
:  (2total
:  (2count
/:-		2Adam/conv2d_45/kernel/m
!:2Adam/conv2d_45/bias/m
/:-2Adam/conv2d_42/kernel/m
!:2Adam/conv2d_42/bias/m
/:-2Adam/conv2d_46/kernel/m
!:2Adam/conv2d_46/bias/m
.:,2"Adam/batch_normalization_7/gamma/m
-:+2!Adam/batch_normalization_7/beta/m
/:-2Adam/conv2d_47/kernel/m
!:2Adam/conv2d_47/bias/m
/:-	2Adam/conv2d_43/kernel/m
!:2Adam/conv2d_43/bias/m
/:-2Adam/conv2d_44/kernel/m
!:2Adam/conv2d_44/bias/m
&:$z2Adam/dense_21/kernel/m
 :2Adam/dense_21/bias/m
&:$2Adam/dense_22/kernel/m
 :2Adam/dense_22/bias/m
&:$2Adam/dense_23/kernel/m
 :2Adam/dense_23/bias/m
1:/2!Adam/prediction_output_0/kernel/m
+:)2Adam/prediction_output_0/bias/m
1:/2!Adam/prediction_output_1/kernel/m
+:)2Adam/prediction_output_1/bias/m
/:-		2Adam/conv2d_45/kernel/v
!:2Adam/conv2d_45/bias/v
/:-2Adam/conv2d_42/kernel/v
!:2Adam/conv2d_42/bias/v
/:-2Adam/conv2d_46/kernel/v
!:2Adam/conv2d_46/bias/v
.:,2"Adam/batch_normalization_7/gamma/v
-:+2!Adam/batch_normalization_7/beta/v
/:-2Adam/conv2d_47/kernel/v
!:2Adam/conv2d_47/bias/v
/:-	2Adam/conv2d_43/kernel/v
!:2Adam/conv2d_43/bias/v
/:-2Adam/conv2d_44/kernel/v
!:2Adam/conv2d_44/bias/v
&:$z2Adam/dense_21/kernel/v
 :2Adam/dense_21/bias/v
&:$2Adam/dense_22/kernel/v
 :2Adam/dense_22/bias/v
&:$2Adam/dense_23/kernel/v
 :2Adam/dense_23/bias/v
1:/2!Adam/prediction_output_0/kernel/v
+:)2Adam/prediction_output_0/bias/v
1:/2!Adam/prediction_output_1/kernel/v
+:)2Adam/prediction_output_1/bias/v╦
!__inference__wrapped_model_716341Ц$#$34FGHI<=YZPQhiЃёІїЊћБцЏюjбg
`б]
[џX
*і'
input_15         <
*і'
input_16         <<
ф "љфї
D
prediction_output_0-і*
prediction_output_0         
D
prediction_output_1-і*
prediction_output_1         В
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_717837ќFGHIMбJ
Cб@
:і7
inputs+                           
p 
ф "?б<
5і2
0+                           
џ В
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_717855ќFGHIMбJ
Cб@
:і7
inputs+                           
p
ф "?б<
5і2
0+                           
џ ─
6__inference_batch_normalization_7_layer_call_fn_717806ЅFGHIMбJ
Cб@
:і7
inputs+                           
p 
ф "2і/+                           ─
6__inference_batch_normalization_7_layer_call_fn_717819ЅFGHIMбJ
Cб@
:і7
inputs+                           
p
ф "2і/+                           Л
I__inference_concatenate_7_layer_call_and_return_conditional_losses_717961ЃZбW
PбM
KџH
"і
inputs/0         
"і
inputs/1         x
ф "%б"
і
0         z
џ е
.__inference_concatenate_7_layer_call_fn_717954vZбW
PбM
KџH
"і
inputs/0         
"і
inputs/1         x
ф "і         zх
E__inference_conv2d_42_layer_call_and_return_conditional_losses_717773l347б4
-б*
(і%
inputs         <
ф "-б*
#і 
0         <
џ Ї
*__inference_conv2d_42_layer_call_fn_717762_347б4
-б*
(і%
inputs         <
ф " і         <х
E__inference_conv2d_43_layer_call_and_return_conditional_losses_717895lYZ7б4
-б*
(і%
inputs         <
ф "-б*
#і 
0         <
џ Ї
*__inference_conv2d_43_layer_call_fn_717884_YZ7б4
-б*
(і%
inputs         <
ф " і         <х
E__inference_conv2d_44_layer_call_and_return_conditional_losses_717926lhi7б4
-б*
(і%
inputs         <
ф "-б*
#і 
0         <
џ Ї
*__inference_conv2d_44_layer_call_fn_717915_hi7б4
-б*
(і%
inputs         <
ф " і         <х
E__inference_conv2d_45_layer_call_and_return_conditional_losses_717715l#$7б4
-б*
(і%
inputs         <<
ф "-б*
#і 
0         
џ Ї
*__inference_conv2d_45_layer_call_fn_717704_#$7б4
-б*
(і%
inputs         <<
ф " і         х
E__inference_conv2d_46_layer_call_and_return_conditional_losses_717793l<=7б4
-б*
(і%
inputs         
ф "-б*
#і 
0         
џ Ї
*__inference_conv2d_46_layer_call_fn_717782_<=7б4
-б*
(і%
inputs         
ф " і         х
E__inference_conv2d_47_layer_call_and_return_conditional_losses_717875lPQ7б4
-б*
(і%
inputs         
ф "-б*
#і 
0         
џ Ї
*__inference_conv2d_47_layer_call_fn_717864_PQ7б4
-б*
(і%
inputs         
ф " і         д
D__inference_dense_21_layer_call_and_return_conditional_losses_717981^Ѓё/б,
%б"
 і
inputs         z
ф "%б"
і
0         
џ ~
)__inference_dense_21_layer_call_fn_717970QЃё/б,
%б"
 і
inputs         z
ф "і         д
D__inference_dense_22_layer_call_and_return_conditional_losses_718001^Ії/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ ~
)__inference_dense_22_layer_call_fn_717990QІї/б,
%б"
 і
inputs         
ф "і         д
D__inference_dense_23_layer_call_and_return_conditional_losses_718021^Њћ/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ ~
)__inference_dense_23_layer_call_fn_718010QЊћ/б,
%б"
 і
inputs         
ф "і         ф
F__inference_flatten_14_layer_call_and_return_conditional_losses_717948`7б4
-б*
(і%
inputs         <
ф "%б"
і
0         x
џ ѓ
+__inference_flatten_14_layer_call_fn_717942S7б4
-б*
(і%
inputs         <
ф "і         xб
F__inference_flatten_15_layer_call_and_return_conditional_losses_717937X/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ z
+__inference_flatten_15_layer_call_fn_717931K/б,
%б"
 і
inputs         
ф "і         █
R__inference_global_max_pooling2d_7_layer_call_and_return_conditional_losses_717906ёRбO
HбE
Cі@
inputs4                                    
ф ".б+
$і!
0                  
џ ▓
7__inference_global_max_pooling2d_7_layer_call_fn_717900wRбO
HбE
Cі@
inputs4                                    
ф "!і                  │
G__inference_joint_model_layer_call_and_return_conditional_losses_717206у$#$34FGHI<=YZPQhiЃёІїЊћБцЏюrбo
hбe
[џX
*і'
input_15         <
*і'
input_16         <<
p 

 
ф "KбH
Aџ>
і
0/0         
і
0/1         
џ │
G__inference_joint_model_layer_call_and_return_conditional_losses_717281у$#$34FGHI<=YZPQhiЃёІїЊћБцЏюrбo
hбe
[џX
*і'
input_15         <
*і'
input_16         <<
p

 
ф "KбH
Aџ>
і
0/0         
і
0/1         
џ │
G__inference_joint_model_layer_call_and_return_conditional_losses_717573у$#$34FGHI<=YZPQhiЃёІїЊћБцЏюrбo
hбe
[џX
*і'
inputs/0         <
*і'
inputs/1         <<
p 

 
ф "KбH
Aџ>
і
0/0         
і
0/1         
џ │
G__inference_joint_model_layer_call_and_return_conditional_losses_717695у$#$34FGHI<=YZPQhiЃёІїЊћБцЏюrбo
hбe
[џX
*і'
inputs/0         <
*і'
inputs/1         <<
p

 
ф "KбH
Aџ>
і
0/0         
і
0/1         
џ і
,__inference_joint_model_layer_call_fn_716747┘$#$34FGHI<=YZPQhiЃёІїЊћБцЏюrбo
hбe
[џX
*і'
input_15         <
*і'
input_16         <<
p 

 
ф "=џ:
і
0         
і
1         і
,__inference_joint_model_layer_call_fn_717131┘$#$34FGHI<=YZPQhiЃёІїЊћБцЏюrбo
hбe
[џX
*і'
input_15         <
*і'
input_16         <<
p

 
ф "=џ:
і
0         
і
1         і
,__inference_joint_model_layer_call_fn_717409┘$#$34FGHI<=YZPQhiЃёІїЊћБцЏюrбo
hбe
[џX
*і'
inputs/0         <
*і'
inputs/1         <<
p 

 
ф "=џ:
і
0         
і
1         і
,__inference_joint_model_layer_call_fn_717469┘$#$34FGHI<=YZPQhiЃёІїЊћБцЏюrбo
hбe
[џX
*і'
inputs/0         <
*і'
inputs/1         <<
p

 
ф "=џ:
і
0         
і
1         ▒
O__inference_prediction_output_0_layer_call_and_return_conditional_losses_718040^Џю/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ Ѕ
4__inference_prediction_output_0_layer_call_fn_718030QЏю/б,
%б"
 і
inputs         
ф "і         ▒
O__inference_prediction_output_1_layer_call_and_return_conditional_losses_718059^Бц/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ Ѕ
4__inference_prediction_output_1_layer_call_fn_718049QБц/б,
%б"
 і
inputs         
ф "і         р
$__inference_signature_wrapper_717349И$#$34FGHI<=YZPQhiЃёІїЊћБцЏю}бz
б 
sфp
6
input_15*і'
input_15         <
6
input_16*і'
input_16         <<"љфї
D
prediction_output_0-і*
prediction_output_0         
D
prediction_output_1-і*
prediction_output_1         Ш
O__inference_spatial_dropout2d_7_layer_call_and_return_conditional_losses_717730бVбS
LбI
Cі@
inputs4                                    
p 
ф "HбE
>і;
04                                    
џ Ш
O__inference_spatial_dropout2d_7_layer_call_and_return_conditional_losses_717753бVбS
LбI
Cі@
inputs4                                    
p
ф "HбE
>і;
04                                    
џ ╬
4__inference_spatial_dropout2d_7_layer_call_fn_717720ЋVбS
LбI
Cі@
inputs4                                    
p 
ф ";і84                                    ╬
4__inference_spatial_dropout2d_7_layer_call_fn_717725ЋVбS
LбI
Cі@
inputs4                                    
p
ф ";і84                                    
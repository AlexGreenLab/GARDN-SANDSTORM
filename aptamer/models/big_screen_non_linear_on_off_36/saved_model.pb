оУ
╙╢
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
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
Ы
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
√
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
epsilonfloat%╖╤8"&
exponential_avg_factorfloat%  А?";
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
М
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
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
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
executor_typestring Ии
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.10.02unknown8╛Ё
Ц
Adam/prediction_output_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/prediction_output_0/bias/v
П
3Adam/prediction_output_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/prediction_output_0/bias/v*
_output_shapes
:*
dtype0
Ю
!Adam/prediction_output_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/prediction_output_0/kernel/v
Ч
5Adam/prediction_output_0/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/prediction_output_0/kernel/v*
_output_shapes

:*
dtype0
В
Adam/dense_215/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_215/bias/v
{
)Adam/dense_215/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_215/bias/v*
_output_shapes
:*
dtype0
К
Adam/dense_215/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_215/kernel/v
Г
+Adam/dense_215/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_215/kernel/v*
_output_shapes

:*
dtype0
В
Adam/dense_214/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_214/bias/v
{
)Adam/dense_214/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_214/bias/v*
_output_shapes
:*
dtype0
К
Adam/dense_214/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_214/kernel/v
Г
+Adam/dense_214/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_214/kernel/v*
_output_shapes

:*
dtype0
В
Adam/dense_213/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_213/bias/v
{
)Adam/dense_213/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_213/bias/v*
_output_shapes
:*
dtype0
Л
Adam/dense_213/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ф*(
shared_nameAdam/dense_213/kernel/v
Д
+Adam/dense_213/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_213/kernel/v*
_output_shapes
:	Ф*
dtype0
Д
Adam/conv2d_428/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_428/bias/v
}
*Adam/conv2d_428/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_428/bias/v*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_428/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_428/kernel/v
Н
,Adam/conv2d_428/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_428/kernel/v*&
_output_shapes
:*
dtype0
Д
Adam/conv2d_427/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_427/bias/v
}
*Adam/conv2d_427/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_427/bias/v*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_427/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/conv2d_427/kernel/v
Н
,Adam/conv2d_427/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_427/kernel/v*&
_output_shapes
:	*
dtype0
Д
Adam/conv2d_431/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_431/bias/v
}
*Adam/conv2d_431/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_431/bias/v*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_431/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_431/kernel/v
Н
,Adam/conv2d_431/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_431/kernel/v*&
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_71/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_71/beta/v
Х
6Adam/batch_normalization_71/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_71/beta/v*
_output_shapes
:*
dtype0
Ю
#Adam/batch_normalization_71/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_71/gamma/v
Ч
7Adam/batch_normalization_71/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_71/gamma/v*
_output_shapes
:*
dtype0
Д
Adam/conv2d_430/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_430/bias/v
}
*Adam/conv2d_430/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_430/bias/v*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_430/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_430/kernel/v
Н
,Adam/conv2d_430/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_430/kernel/v*&
_output_shapes
:*
dtype0
Д
Adam/conv2d_426/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_426/bias/v
}
*Adam/conv2d_426/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_426/bias/v*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_426/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_426/kernel/v
Н
,Adam/conv2d_426/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_426/kernel/v*&
_output_shapes
:*
dtype0
Д
Adam/conv2d_429/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_429/bias/v
}
*Adam/conv2d_429/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_429/bias/v*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_429/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*)
shared_nameAdam/conv2d_429/kernel/v
Н
,Adam/conv2d_429/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_429/kernel/v*&
_output_shapes
:		*
dtype0
Ц
Adam/prediction_output_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/prediction_output_0/bias/m
П
3Adam/prediction_output_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/prediction_output_0/bias/m*
_output_shapes
:*
dtype0
Ю
!Adam/prediction_output_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/prediction_output_0/kernel/m
Ч
5Adam/prediction_output_0/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/prediction_output_0/kernel/m*
_output_shapes

:*
dtype0
В
Adam/dense_215/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_215/bias/m
{
)Adam/dense_215/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_215/bias/m*
_output_shapes
:*
dtype0
К
Adam/dense_215/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_215/kernel/m
Г
+Adam/dense_215/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_215/kernel/m*
_output_shapes

:*
dtype0
В
Adam/dense_214/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_214/bias/m
{
)Adam/dense_214/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_214/bias/m*
_output_shapes
:*
dtype0
К
Adam/dense_214/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_214/kernel/m
Г
+Adam/dense_214/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_214/kernel/m*
_output_shapes

:*
dtype0
В
Adam/dense_213/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_213/bias/m
{
)Adam/dense_213/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_213/bias/m*
_output_shapes
:*
dtype0
Л
Adam/dense_213/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ф*(
shared_nameAdam/dense_213/kernel/m
Д
+Adam/dense_213/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_213/kernel/m*
_output_shapes
:	Ф*
dtype0
Д
Adam/conv2d_428/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_428/bias/m
}
*Adam/conv2d_428/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_428/bias/m*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_428/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_428/kernel/m
Н
,Adam/conv2d_428/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_428/kernel/m*&
_output_shapes
:*
dtype0
Д
Adam/conv2d_427/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_427/bias/m
}
*Adam/conv2d_427/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_427/bias/m*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_427/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/conv2d_427/kernel/m
Н
,Adam/conv2d_427/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_427/kernel/m*&
_output_shapes
:	*
dtype0
Д
Adam/conv2d_431/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_431/bias/m
}
*Adam/conv2d_431/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_431/bias/m*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_431/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_431/kernel/m
Н
,Adam/conv2d_431/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_431/kernel/m*&
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_71/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_71/beta/m
Х
6Adam/batch_normalization_71/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_71/beta/m*
_output_shapes
:*
dtype0
Ю
#Adam/batch_normalization_71/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_71/gamma/m
Ч
7Adam/batch_normalization_71/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_71/gamma/m*
_output_shapes
:*
dtype0
Д
Adam/conv2d_430/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_430/bias/m
}
*Adam/conv2d_430/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_430/bias/m*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_430/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_430/kernel/m
Н
,Adam/conv2d_430/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_430/kernel/m*&
_output_shapes
:*
dtype0
Д
Adam/conv2d_426/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_426/bias/m
}
*Adam/conv2d_426/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_426/bias/m*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_426/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_426/kernel/m
Н
,Adam/conv2d_426/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_426/kernel/m*&
_output_shapes
:*
dtype0
Д
Adam/conv2d_429/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_429/bias/m
}
*Adam/conv2d_429/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_429/bias/m*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_429/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*)
shared_nameAdam/conv2d_429/kernel/m
Н
,Adam/conv2d_429/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_429/kernel/m*&
_output_shapes
:		*
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
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
И
prediction_output_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameprediction_output_0/bias
Б
,prediction_output_0/bias/Read/ReadVariableOpReadVariableOpprediction_output_0/bias*
_output_shapes
:*
dtype0
Р
prediction_output_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameprediction_output_0/kernel
Й
.prediction_output_0/kernel/Read/ReadVariableOpReadVariableOpprediction_output_0/kernel*
_output_shapes

:*
dtype0
t
dense_215/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_215/bias
m
"dense_215/bias/Read/ReadVariableOpReadVariableOpdense_215/bias*
_output_shapes
:*
dtype0
|
dense_215/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_215/kernel
u
$dense_215/kernel/Read/ReadVariableOpReadVariableOpdense_215/kernel*
_output_shapes

:*
dtype0
t
dense_214/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_214/bias
m
"dense_214/bias/Read/ReadVariableOpReadVariableOpdense_214/bias*
_output_shapes
:*
dtype0
|
dense_214/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_214/kernel
u
$dense_214/kernel/Read/ReadVariableOpReadVariableOpdense_214/kernel*
_output_shapes

:*
dtype0
t
dense_213/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_213/bias
m
"dense_213/bias/Read/ReadVariableOpReadVariableOpdense_213/bias*
_output_shapes
:*
dtype0
}
dense_213/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ф*!
shared_namedense_213/kernel
v
$dense_213/kernel/Read/ReadVariableOpReadVariableOpdense_213/kernel*
_output_shapes
:	Ф*
dtype0
v
conv2d_428/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_428/bias
o
#conv2d_428/bias/Read/ReadVariableOpReadVariableOpconv2d_428/bias*
_output_shapes
:*
dtype0
Ж
conv2d_428/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_428/kernel

%conv2d_428/kernel/Read/ReadVariableOpReadVariableOpconv2d_428/kernel*&
_output_shapes
:*
dtype0
v
conv2d_427/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_427/bias
o
#conv2d_427/bias/Read/ReadVariableOpReadVariableOpconv2d_427/bias*
_output_shapes
:*
dtype0
Ж
conv2d_427/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*"
shared_nameconv2d_427/kernel

%conv2d_427/kernel/Read/ReadVariableOpReadVariableOpconv2d_427/kernel*&
_output_shapes
:	*
dtype0
v
conv2d_431/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_431/bias
o
#conv2d_431/bias/Read/ReadVariableOpReadVariableOpconv2d_431/bias*
_output_shapes
:*
dtype0
Ж
conv2d_431/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_431/kernel

%conv2d_431/kernel/Read/ReadVariableOpReadVariableOpconv2d_431/kernel*&
_output_shapes
:*
dtype0
д
&batch_normalization_71/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_71/moving_variance
Э
:batch_normalization_71/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_71/moving_variance*
_output_shapes
:*
dtype0
Ь
"batch_normalization_71/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_71/moving_mean
Х
6batch_normalization_71/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_71/moving_mean*
_output_shapes
:*
dtype0
О
batch_normalization_71/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_71/beta
З
/batch_normalization_71/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_71/beta*
_output_shapes
:*
dtype0
Р
batch_normalization_71/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_71/gamma
Й
0batch_normalization_71/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_71/gamma*
_output_shapes
:*
dtype0
v
conv2d_430/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_430/bias
o
#conv2d_430/bias/Read/ReadVariableOpReadVariableOpconv2d_430/bias*
_output_shapes
:*
dtype0
Ж
conv2d_430/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_430/kernel

%conv2d_430/kernel/Read/ReadVariableOpReadVariableOpconv2d_430/kernel*&
_output_shapes
:*
dtype0
v
conv2d_426/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_426/bias
o
#conv2d_426/bias/Read/ReadVariableOpReadVariableOpconv2d_426/bias*
_output_shapes
:*
dtype0
Ж
conv2d_426/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_426/kernel

%conv2d_426/kernel/Read/ReadVariableOpReadVariableOpconv2d_426/kernel*&
_output_shapes
:*
dtype0
v
conv2d_429/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_429/bias
o
#conv2d_429/bias/Read/ReadVariableOpReadVariableOpconv2d_429/bias*
_output_shapes
:*
dtype0
Ж
conv2d_429/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*"
shared_nameconv2d_429/kernel

%conv2d_429/kernel/Read/ReadVariableOpReadVariableOpconv2d_429/kernel*&
_output_shapes
:		*
dtype0
О
serving_default_input_143Placeholder*0
_output_shapes
:         Й*
dtype0*%
shape:         Й
Р
serving_default_input_144Placeholder*1
_output_shapes
:         ЙЙ*
dtype0*&
shape:         ЙЙ
 
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_143serving_default_input_144conv2d_429/kernelconv2d_429/biasconv2d_426/kernelconv2d_426/biasbatch_normalization_71/gammabatch_normalization_71/beta"batch_normalization_71/moving_mean&batch_normalization_71/moving_varianceconv2d_430/kernelconv2d_430/biasconv2d_427/kernelconv2d_427/biasconv2d_431/kernelconv2d_431/biasconv2d_428/kernelconv2d_428/biasdense_213/kerneldense_213/biasdense_214/kerneldense_214/biasdense_215/kerneldense_215/biasprediction_output_0/kernelprediction_output_0/bias*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *.
f)R'
%__inference_signature_wrapper_4115516

NoOpNoOp
єа
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*на
valueваBЮа BЦа
▌
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
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
╚
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias
 $_jit_compiled_convolution_op*
* 
е
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
+_random_generator* 
╚
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias
 4_jit_compiled_convolution_op*
╚
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias
 =_jit_compiled_convolution_op*
╒
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses
Daxis
	Egamma
Fbeta
Gmoving_mean
Hmoving_variance*
╚
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses

Okernel
Pbias
 Q_jit_compiled_convolution_op*
╚
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses

Xkernel
Ybias
 Z_jit_compiled_convolution_op*
О
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses* 
╚
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses

gkernel
hbias
 i_jit_compiled_convolution_op*
О
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses* 
О
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses* 
О
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses* 
к
|	variables
}trainable_variables
~regularization_losses
	keras_api
А__call__
+Б&call_and_return_all_conditional_losses
Вkernel
	Гbias*
о
Д	variables
Еtrainable_variables
Жregularization_losses
З	keras_api
И__call__
+Й&call_and_return_all_conditional_losses
Кkernel
	Лbias*
о
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
Р__call__
+С&call_and_return_all_conditional_losses
Тkernel
	Уbias*
о
Ф	variables
Хtrainable_variables
Цregularization_losses
Ч	keras_api
Ш__call__
+Щ&call_and_return_all_conditional_losses
Ъkernel
	Ыbias*
┬
"0
#1
22
33
;4
<5
E6
F7
G8
H9
O10
P11
X12
Y13
g14
h15
В16
Г17
К18
Л19
Т20
У21
Ъ22
Ы23*
▓
"0
#1
22
33
;4
<5
E6
F7
O8
P9
X10
Y11
g12
h13
В14
Г15
К16
Л17
Т18
У19
Ъ20
Ы21*
* 
╡
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
бtrace_0
вtrace_1
гtrace_2
дtrace_3* 
:
еtrace_0
жtrace_1
зtrace_2
иtrace_3* 
* 
С
йbeta_1
кbeta_2

лdecay
мlearning_rate
	нiter"mи#mй2mк3mл;mм<mнEmоFmпOm░Pm▒Xm▓Ym│gm┤hm╡	Вm╢	Гm╖	Кm╕	Лm╣	Тm║	Уm╗	Ъm╝	Ыm╜"v╛#v┐2v└3v┴;v┬<v├Ev─Fv┼Ov╞Pv╟Xv╚Yv╔gv╩hv╦	Вv╠	Гv═	Кv╬	Лv╧	Тv╨	Уv╤	Ъv╥	Ыv╙*

оserving_default* 

"0
#1*

"0
#1*
* 
Ш
пnon_trainable_variables
░layers
▒metrics
 ▓layer_regularization_losses
│layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

┤trace_0* 

╡trace_0* 
a[
VARIABLE_VALUEconv2d_429/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_429/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
╢non_trainable_variables
╖layers
╕metrics
 ╣layer_regularization_losses
║layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses* 

╗trace_0
╝trace_1* 

╜trace_0
╛trace_1* 
* 

20
31*

20
31*
* 
Ш
┐non_trainable_variables
└layers
┴metrics
 ┬layer_regularization_losses
├layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*

─trace_0* 

┼trace_0* 
a[
VARIABLE_VALUEconv2d_426/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_426/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

;0
<1*

;0
<1*
* 
Ш
╞non_trainable_variables
╟layers
╚metrics
 ╔layer_regularization_losses
╩layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

╦trace_0* 

╠trace_0* 
a[
VARIABLE_VALUEconv2d_430/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_430/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
E0
F1
G2
H3*

E0
F1*
* 
Ш
═non_trainable_variables
╬layers
╧metrics
 ╨layer_regularization_losses
╤layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*

╥trace_0
╙trace_1* 

╘trace_0
╒trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_71/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_71/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_71/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_71/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

O0
P1*

O0
P1*
* 
Ш
╓non_trainable_variables
╫layers
╪metrics
 ┘layer_regularization_losses
┌layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses*

█trace_0* 

▄trace_0* 
a[
VARIABLE_VALUEconv2d_431/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_431/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

X0
Y1*

X0
Y1*
* 
Ш
▌non_trainable_variables
▐layers
▀metrics
 рlayer_regularization_losses
сlayer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses*

тtrace_0* 

уtrace_0* 
a[
VARIABLE_VALUEconv2d_427/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_427/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
фnon_trainable_variables
хlayers
цmetrics
 чlayer_regularization_losses
шlayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses* 

щtrace_0* 

ъtrace_0* 

g0
h1*

g0
h1*
* 
Ш
ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
яlayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*

Ёtrace_0* 

ёtrace_0* 
a[
VARIABLE_VALUEconv2d_428/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_428/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
Єnon_trainable_variables
єlayers
Їmetrics
 їlayer_regularization_losses
Ўlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses* 

ўtrace_0* 

°trace_0* 
* 
* 
* 
Ц
∙non_trainable_variables
·layers
√metrics
 №layer_regularization_losses
¤layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses* 

■trace_0* 

 trace_0* 
* 
* 
* 
Ц
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses* 

Еtrace_0* 

Жtrace_0* 

В0
Г1*

В0
Г1*
* 
Ы
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
|	variables
}trainable_variables
~regularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses*

Мtrace_0* 

Нtrace_0* 
`Z
VARIABLE_VALUEdense_213/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_213/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

К0
Л1*

К0
Л1*
* 
Ю
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
Д	variables
Еtrainable_variables
Жregularization_losses
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses*

Уtrace_0* 

Фtrace_0* 
`Z
VARIABLE_VALUEdense_214/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_214/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

Т0
У1*

Т0
У1*
* 
Ю
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
М	variables
Нtrainable_variables
Оregularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses*

Ъtrace_0* 

Ыtrace_0* 
`Z
VARIABLE_VALUEdense_215/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_215/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

Ъ0
Ы1*

Ъ0
Ы1*
* 
Ю
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
Ф	variables
Хtrainable_variables
Цregularization_losses
Ш__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses*

бtrace_0* 

вtrace_0* 
ke
VARIABLE_VALUEprediction_output_0/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEprediction_output_0/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*

G0
H1*
К
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
17*

г0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
KE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
G0
H1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
д	variables
е	keras_api

жtotal

зcount*

ж0
з1*

д	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_429/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_429/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_426/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_426/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_430/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_430/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/batch_normalization_71/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/batch_normalization_71/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_431/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_431/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_427/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_427/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_428/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_428/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdam/dense_213/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_213/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdam/dense_214/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_214/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdam/dense_215/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_215/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE!Adam/prediction_output_0/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЛД
VARIABLE_VALUEAdam/prediction_output_0/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_429/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_429/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_426/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_426/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_430/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_430/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/batch_normalization_71/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/batch_normalization_71/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_431/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_431/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_427/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_427/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_428/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_428/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdam/dense_213/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_213/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdam/dense_214/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_214/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdam/dense_215/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_215/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE!Adam/prediction_output_0/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЛД
VARIABLE_VALUEAdam/prediction_output_0/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Т
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%conv2d_429/kernel/Read/ReadVariableOp#conv2d_429/bias/Read/ReadVariableOp%conv2d_426/kernel/Read/ReadVariableOp#conv2d_426/bias/Read/ReadVariableOp%conv2d_430/kernel/Read/ReadVariableOp#conv2d_430/bias/Read/ReadVariableOp0batch_normalization_71/gamma/Read/ReadVariableOp/batch_normalization_71/beta/Read/ReadVariableOp6batch_normalization_71/moving_mean/Read/ReadVariableOp:batch_normalization_71/moving_variance/Read/ReadVariableOp%conv2d_431/kernel/Read/ReadVariableOp#conv2d_431/bias/Read/ReadVariableOp%conv2d_427/kernel/Read/ReadVariableOp#conv2d_427/bias/Read/ReadVariableOp%conv2d_428/kernel/Read/ReadVariableOp#conv2d_428/bias/Read/ReadVariableOp$dense_213/kernel/Read/ReadVariableOp"dense_213/bias/Read/ReadVariableOp$dense_214/kernel/Read/ReadVariableOp"dense_214/bias/Read/ReadVariableOp$dense_215/kernel/Read/ReadVariableOp"dense_215/bias/Read/ReadVariableOp.prediction_output_0/kernel/Read/ReadVariableOp,prediction_output_0/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp,Adam/conv2d_429/kernel/m/Read/ReadVariableOp*Adam/conv2d_429/bias/m/Read/ReadVariableOp,Adam/conv2d_426/kernel/m/Read/ReadVariableOp*Adam/conv2d_426/bias/m/Read/ReadVariableOp,Adam/conv2d_430/kernel/m/Read/ReadVariableOp*Adam/conv2d_430/bias/m/Read/ReadVariableOp7Adam/batch_normalization_71/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_71/beta/m/Read/ReadVariableOp,Adam/conv2d_431/kernel/m/Read/ReadVariableOp*Adam/conv2d_431/bias/m/Read/ReadVariableOp,Adam/conv2d_427/kernel/m/Read/ReadVariableOp*Adam/conv2d_427/bias/m/Read/ReadVariableOp,Adam/conv2d_428/kernel/m/Read/ReadVariableOp*Adam/conv2d_428/bias/m/Read/ReadVariableOp+Adam/dense_213/kernel/m/Read/ReadVariableOp)Adam/dense_213/bias/m/Read/ReadVariableOp+Adam/dense_214/kernel/m/Read/ReadVariableOp)Adam/dense_214/bias/m/Read/ReadVariableOp+Adam/dense_215/kernel/m/Read/ReadVariableOp)Adam/dense_215/bias/m/Read/ReadVariableOp5Adam/prediction_output_0/kernel/m/Read/ReadVariableOp3Adam/prediction_output_0/bias/m/Read/ReadVariableOp,Adam/conv2d_429/kernel/v/Read/ReadVariableOp*Adam/conv2d_429/bias/v/Read/ReadVariableOp,Adam/conv2d_426/kernel/v/Read/ReadVariableOp*Adam/conv2d_426/bias/v/Read/ReadVariableOp,Adam/conv2d_430/kernel/v/Read/ReadVariableOp*Adam/conv2d_430/bias/v/Read/ReadVariableOp7Adam/batch_normalization_71/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_71/beta/v/Read/ReadVariableOp,Adam/conv2d_431/kernel/v/Read/ReadVariableOp*Adam/conv2d_431/bias/v/Read/ReadVariableOp,Adam/conv2d_427/kernel/v/Read/ReadVariableOp*Adam/conv2d_427/bias/v/Read/ReadVariableOp,Adam/conv2d_428/kernel/v/Read/ReadVariableOp*Adam/conv2d_428/bias/v/Read/ReadVariableOp+Adam/dense_213/kernel/v/Read/ReadVariableOp)Adam/dense_213/bias/v/Read/ReadVariableOp+Adam/dense_214/kernel/v/Read/ReadVariableOp)Adam/dense_214/bias/v/Read/ReadVariableOp+Adam/dense_215/kernel/v/Read/ReadVariableOp)Adam/dense_215/bias/v/Read/ReadVariableOp5Adam/prediction_output_0/kernel/v/Read/ReadVariableOp3Adam/prediction_output_0/bias/v/Read/ReadVariableOpConst*X
TinQ
O2M	*
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
GPU2*0J 8В *)
f$R"
 __inference__traced_save_4116430
▒
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_429/kernelconv2d_429/biasconv2d_426/kernelconv2d_426/biasconv2d_430/kernelconv2d_430/biasbatch_normalization_71/gammabatch_normalization_71/beta"batch_normalization_71/moving_mean&batch_normalization_71/moving_varianceconv2d_431/kernelconv2d_431/biasconv2d_427/kernelconv2d_427/biasconv2d_428/kernelconv2d_428/biasdense_213/kerneldense_213/biasdense_214/kerneldense_214/biasdense_215/kerneldense_215/biasprediction_output_0/kernelprediction_output_0/biasbeta_1beta_2decaylearning_rate	Adam/itertotalcountAdam/conv2d_429/kernel/mAdam/conv2d_429/bias/mAdam/conv2d_426/kernel/mAdam/conv2d_426/bias/mAdam/conv2d_430/kernel/mAdam/conv2d_430/bias/m#Adam/batch_normalization_71/gamma/m"Adam/batch_normalization_71/beta/mAdam/conv2d_431/kernel/mAdam/conv2d_431/bias/mAdam/conv2d_427/kernel/mAdam/conv2d_427/bias/mAdam/conv2d_428/kernel/mAdam/conv2d_428/bias/mAdam/dense_213/kernel/mAdam/dense_213/bias/mAdam/dense_214/kernel/mAdam/dense_214/bias/mAdam/dense_215/kernel/mAdam/dense_215/bias/m!Adam/prediction_output_0/kernel/mAdam/prediction_output_0/bias/mAdam/conv2d_429/kernel/vAdam/conv2d_429/bias/vAdam/conv2d_426/kernel/vAdam/conv2d_426/bias/vAdam/conv2d_430/kernel/vAdam/conv2d_430/bias/v#Adam/batch_normalization_71/gamma/v"Adam/batch_normalization_71/beta/vAdam/conv2d_431/kernel/vAdam/conv2d_431/bias/vAdam/conv2d_427/kernel/vAdam/conv2d_427/bias/vAdam/conv2d_428/kernel/vAdam/conv2d_428/bias/vAdam/dense_213/kernel/vAdam/dense_213/bias/vAdam/dense_214/kernel/vAdam/dense_214/bias/vAdam/dense_215/kernel/vAdam/dense_215/bias/v!Adam/prediction_output_0/kernel/vAdam/prediction_output_0/bias/v*W
TinP
N2L*
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
GPU2*0J 8В *,
f'R%
#__inference__traced_restore_4116665▒¤
░
p
T__inference_global_max_pooling2d_71_layer_call_and_return_conditional_losses_4114697

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
Н
А
G__inference_conv2d_431_layer_call_and_return_conditional_losses_4114798

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ы
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides	
Йr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ЙY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Йj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Йw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Й: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         Й
 
_user_specified_nameinputs
╒
▓
-__inference_joint_model_layer_call_fn_4115624
inputs_0
inputs_1!
unknown:		
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:	

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:	Ф

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identityИвStatefulPartitionedCallЦ
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
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *8
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_joint_model_layer_call_and_return_conditional_losses_4115211o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:         Й:         ЙЙ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:         Й
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:         ЙЙ
"
_user_specified_name
inputs/1
еy
Р
H__inference_joint_model_layer_call_and_return_conditional_losses_4115721
inputs_0
inputs_1C
)conv2d_429_conv2d_readvariableop_resource:		8
*conv2d_429_biasadd_readvariableop_resource:C
)conv2d_426_conv2d_readvariableop_resource:8
*conv2d_426_biasadd_readvariableop_resource:<
.batch_normalization_71_readvariableop_resource:>
0batch_normalization_71_readvariableop_1_resource:M
?batch_normalization_71_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_71_fusedbatchnormv3_readvariableop_1_resource:C
)conv2d_430_conv2d_readvariableop_resource:8
*conv2d_430_biasadd_readvariableop_resource:C
)conv2d_427_conv2d_readvariableop_resource:	8
*conv2d_427_biasadd_readvariableop_resource:C
)conv2d_431_conv2d_readvariableop_resource:8
*conv2d_431_biasadd_readvariableop_resource:C
)conv2d_428_conv2d_readvariableop_resource:8
*conv2d_428_biasadd_readvariableop_resource:;
(dense_213_matmul_readvariableop_resource:	Ф7
)dense_213_biasadd_readvariableop_resource::
(dense_214_matmul_readvariableop_resource:7
)dense_214_biasadd_readvariableop_resource::
(dense_215_matmul_readvariableop_resource:7
)dense_215_biasadd_readvariableop_resource:D
2prediction_output_0_matmul_readvariableop_resource:A
3prediction_output_0_biasadd_readvariableop_resource:
identityИв6batch_normalization_71/FusedBatchNormV3/ReadVariableOpв8batch_normalization_71/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_71/ReadVariableOpв'batch_normalization_71/ReadVariableOp_1в!conv2d_426/BiasAdd/ReadVariableOpв conv2d_426/Conv2D/ReadVariableOpв!conv2d_427/BiasAdd/ReadVariableOpв conv2d_427/Conv2D/ReadVariableOpв!conv2d_428/BiasAdd/ReadVariableOpв conv2d_428/Conv2D/ReadVariableOpв!conv2d_429/BiasAdd/ReadVariableOpв conv2d_429/Conv2D/ReadVariableOpв!conv2d_430/BiasAdd/ReadVariableOpв conv2d_430/Conv2D/ReadVariableOpв!conv2d_431/BiasAdd/ReadVariableOpв conv2d_431/Conv2D/ReadVariableOpв dense_213/BiasAdd/ReadVariableOpвdense_213/MatMul/ReadVariableOpв dense_214/BiasAdd/ReadVariableOpвdense_214/MatMul/ReadVariableOpв dense_215/BiasAdd/ReadVariableOpвdense_215/MatMul/ReadVariableOpв*prediction_output_0/BiasAdd/ReadVariableOpв)prediction_output_0/MatMul/ReadVariableOpТ
 conv2d_429/Conv2D/ReadVariableOpReadVariableOp)conv2d_429_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0│
conv2d_429/Conv2DConv2Dinputs_1(conv2d_429/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides	
ЙИ
!conv2d_429/BiasAdd/ReadVariableOpReadVariableOp*conv2d_429_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv2d_429/BiasAddBiasAddconv2d_429/Conv2D:output:0)conv2d_429/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Йo
conv2d_429/ReluReluconv2d_429/BiasAdd:output:0*
T0*0
_output_shapes
:         ЙТ
 conv2d_426/Conv2D/ReadVariableOpReadVariableOp)conv2d_426_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0▓
conv2d_426/Conv2DConv2Dinputs_0(conv2d_426/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides
И
!conv2d_426/BiasAdd/ReadVariableOpReadVariableOp*conv2d_426_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv2d_426/BiasAddBiasAddconv2d_426/Conv2D:output:0)conv2d_426/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Йo
conv2d_426/ReluReluconv2d_426/BiasAdd:output:0*
T0*0
_output_shapes
:         ЙГ
spatial_dropout2d_71/IdentityIdentityconv2d_429/Relu:activations:0*
T0*0
_output_shapes
:         ЙР
%batch_normalization_71/ReadVariableOpReadVariableOp.batch_normalization_71_readvariableop_resource*
_output_shapes
:*
dtype0Ф
'batch_normalization_71/ReadVariableOp_1ReadVariableOp0batch_normalization_71_readvariableop_1_resource*
_output_shapes
:*
dtype0▓
6batch_normalization_71/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_71_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╢
8batch_normalization_71/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_71_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0┴
'batch_normalization_71/FusedBatchNormV3FusedBatchNormV3conv2d_426/Relu:activations:0-batch_normalization_71/ReadVariableOp:value:0/batch_normalization_71/ReadVariableOp_1:value:0>batch_normalization_71/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_71/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         Й:::::*
epsilon%oГ:*
is_training( Т
 conv2d_430/Conv2D/ReadVariableOpReadVariableOp)conv2d_430_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╤
conv2d_430/Conv2DConv2D&spatial_dropout2d_71/Identity:output:0(conv2d_430/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides	
ЙИ
!conv2d_430/BiasAdd/ReadVariableOpReadVariableOp*conv2d_430_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv2d_430/BiasAddBiasAddconv2d_430/Conv2D:output:0)conv2d_430/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Йo
conv2d_430/ReluReluconv2d_430/BiasAdd:output:0*
T0*0
_output_shapes
:         ЙТ
 conv2d_427/Conv2D/ReadVariableOpReadVariableOp)conv2d_427_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0╒
conv2d_427/Conv2DConv2D+batch_normalization_71/FusedBatchNormV3:y:0(conv2d_427/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides
И
!conv2d_427/BiasAdd/ReadVariableOpReadVariableOp*conv2d_427_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv2d_427/BiasAddBiasAddconv2d_427/Conv2D:output:0)conv2d_427/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Йo
conv2d_427/ReluReluconv2d_427/BiasAdd:output:0*
T0*0
_output_shapes
:         ЙТ
 conv2d_431/Conv2D/ReadVariableOpReadVariableOp)conv2d_431_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╚
conv2d_431/Conv2DConv2Dconv2d_430/Relu:activations:0(conv2d_431/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides	
ЙИ
!conv2d_431/BiasAdd/ReadVariableOpReadVariableOp*conv2d_431_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv2d_431/BiasAddBiasAddconv2d_431/Conv2D:output:0)conv2d_431/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Йo
conv2d_431/ReluReluconv2d_431/BiasAdd:output:0*
T0*0
_output_shapes
:         ЙТ
 conv2d_428/Conv2D/ReadVariableOpReadVariableOp)conv2d_428_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╟
conv2d_428/Conv2DConv2Dconv2d_427/Relu:activations:0(conv2d_428/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides
И
!conv2d_428/BiasAdd/ReadVariableOpReadVariableOp*conv2d_428_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv2d_428/BiasAddBiasAddconv2d_428/Conv2D:output:0)conv2d_428/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Йo
conv2d_428/ReluReluconv2d_428/BiasAdd:output:0*
T0*0
_output_shapes
:         Й~
-global_max_pooling2d_71/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      л
global_max_pooling2d_71/MaxMaxconv2d_431/Relu:activations:06global_max_pooling2d_71/Max/reduction_indices:output:0*
T0*'
_output_shapes
:         b
flatten_143/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Т
flatten_143/ReshapeReshape$global_max_pooling2d_71/Max:output:0flatten_143/Const:output:0*
T0*'
_output_shapes
:         b
flatten_142/ConstConst*
_output_shapes
:*
dtype0*
valueB"      М
flatten_142/ReshapeReshapeconv2d_428/Relu:activations:0flatten_142/Const:output:0*
T0*(
_output_shapes
:         Т\
concatenate_71/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :╛
concatenate_71/concatConcatV2flatten_143/Reshape:output:0flatten_142/Reshape:output:0#concatenate_71/concat/axis:output:0*
N*
T0*(
_output_shapes
:         ФЙ
dense_213/MatMul/ReadVariableOpReadVariableOp(dense_213_matmul_readvariableop_resource*
_output_shapes
:	Ф*
dtype0Х
dense_213/MatMulMatMulconcatenate_71/concat:output:0'dense_213/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ж
 dense_213/BiasAdd/ReadVariableOpReadVariableOp)dense_213_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
dense_213/BiasAddBiasAdddense_213/MatMul:product:0(dense_213/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d
dense_213/ReluReludense_213/BiasAdd:output:0*
T0*'
_output_shapes
:         И
dense_214/MatMul/ReadVariableOpReadVariableOp(dense_214_matmul_readvariableop_resource*
_output_shapes

:*
dtype0У
dense_214/MatMulMatMuldense_213/Relu:activations:0'dense_214/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ж
 dense_214/BiasAdd/ReadVariableOpReadVariableOp)dense_214_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
dense_214/BiasAddBiasAdddense_214/MatMul:product:0(dense_214/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d
dense_214/ReluReludense_214/BiasAdd:output:0*
T0*'
_output_shapes
:         И
dense_215/MatMul/ReadVariableOpReadVariableOp(dense_215_matmul_readvariableop_resource*
_output_shapes

:*
dtype0У
dense_215/MatMulMatMuldense_214/Relu:activations:0'dense_215/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ж
 dense_215/BiasAdd/ReadVariableOpReadVariableOp)dense_215_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
dense_215/BiasAddBiasAdddense_215/MatMul:product:0(dense_215/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d
dense_215/ReluReludense_215/BiasAdd:output:0*
T0*'
_output_shapes
:         Ь
)prediction_output_0/MatMul/ReadVariableOpReadVariableOp2prediction_output_0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0з
prediction_output_0/MatMulMatMuldense_215/Relu:activations:01prediction_output_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ъ
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
:         ▐
NoOpNoOp7^batch_normalization_71/FusedBatchNormV3/ReadVariableOp9^batch_normalization_71/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_71/ReadVariableOp(^batch_normalization_71/ReadVariableOp_1"^conv2d_426/BiasAdd/ReadVariableOp!^conv2d_426/Conv2D/ReadVariableOp"^conv2d_427/BiasAdd/ReadVariableOp!^conv2d_427/Conv2D/ReadVariableOp"^conv2d_428/BiasAdd/ReadVariableOp!^conv2d_428/Conv2D/ReadVariableOp"^conv2d_429/BiasAdd/ReadVariableOp!^conv2d_429/Conv2D/ReadVariableOp"^conv2d_430/BiasAdd/ReadVariableOp!^conv2d_430/Conv2D/ReadVariableOp"^conv2d_431/BiasAdd/ReadVariableOp!^conv2d_431/Conv2D/ReadVariableOp!^dense_213/BiasAdd/ReadVariableOp ^dense_213/MatMul/ReadVariableOp!^dense_214/BiasAdd/ReadVariableOp ^dense_214/MatMul/ReadVariableOp!^dense_215/BiasAdd/ReadVariableOp ^dense_215/MatMul/ReadVariableOp+^prediction_output_0/BiasAdd/ReadVariableOp*^prediction_output_0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:         Й:         ЙЙ: : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_71/FusedBatchNormV3/ReadVariableOp6batch_normalization_71/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_71/FusedBatchNormV3/ReadVariableOp_18batch_normalization_71/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_71/ReadVariableOp%batch_normalization_71/ReadVariableOp2R
'batch_normalization_71/ReadVariableOp_1'batch_normalization_71/ReadVariableOp_12F
!conv2d_426/BiasAdd/ReadVariableOp!conv2d_426/BiasAdd/ReadVariableOp2D
 conv2d_426/Conv2D/ReadVariableOp conv2d_426/Conv2D/ReadVariableOp2F
!conv2d_427/BiasAdd/ReadVariableOp!conv2d_427/BiasAdd/ReadVariableOp2D
 conv2d_427/Conv2D/ReadVariableOp conv2d_427/Conv2D/ReadVariableOp2F
!conv2d_428/BiasAdd/ReadVariableOp!conv2d_428/BiasAdd/ReadVariableOp2D
 conv2d_428/Conv2D/ReadVariableOp conv2d_428/Conv2D/ReadVariableOp2F
!conv2d_429/BiasAdd/ReadVariableOp!conv2d_429/BiasAdd/ReadVariableOp2D
 conv2d_429/Conv2D/ReadVariableOp conv2d_429/Conv2D/ReadVariableOp2F
!conv2d_430/BiasAdd/ReadVariableOp!conv2d_430/BiasAdd/ReadVariableOp2D
 conv2d_430/Conv2D/ReadVariableOp conv2d_430/Conv2D/ReadVariableOp2F
!conv2d_431/BiasAdd/ReadVariableOp!conv2d_431/BiasAdd/ReadVariableOp2D
 conv2d_431/Conv2D/ReadVariableOp conv2d_431/Conv2D/ReadVariableOp2D
 dense_213/BiasAdd/ReadVariableOp dense_213/BiasAdd/ReadVariableOp2B
dense_213/MatMul/ReadVariableOpdense_213/MatMul/ReadVariableOp2D
 dense_214/BiasAdd/ReadVariableOp dense_214/BiasAdd/ReadVariableOp2B
dense_214/MatMul/ReadVariableOpdense_214/MatMul/ReadVariableOp2D
 dense_215/BiasAdd/ReadVariableOp dense_215/BiasAdd/ReadVariableOp2B
dense_215/MatMul/ReadVariableOpdense_215/MatMul/ReadVariableOp2X
*prediction_output_0/BiasAdd/ReadVariableOp*prediction_output_0/BiasAdd/ReadVariableOp2V
)prediction_output_0/MatMul/ReadVariableOp)prediction_output_0/MatMul/ReadVariableOp:Z V
0
_output_shapes
:         Й
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:         ЙЙ
"
_user_specified_name
inputs/1
╝
I
-__inference_flatten_142_layer_call_fn_4116083

inputs
identity╖
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Т* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_flatten_142_layer_call_and_return_conditional_losses_4114836a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         Т"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         Й:X T
0
_output_shapes
:         Й
 
_user_specified_nameinputs
кP
Ц
H__inference_joint_model_layer_call_and_return_conditional_losses_4115454
	input_143
	input_144,
conv2d_429_4115389:		 
conv2d_429_4115391:,
conv2d_426_4115394: 
conv2d_426_4115396:,
batch_normalization_71_4115400:,
batch_normalization_71_4115402:,
batch_normalization_71_4115404:,
batch_normalization_71_4115406:,
conv2d_430_4115409: 
conv2d_430_4115411:,
conv2d_427_4115414:	 
conv2d_427_4115416:,
conv2d_431_4115419: 
conv2d_431_4115421:,
conv2d_428_4115424: 
conv2d_428_4115426:$
dense_213_4115433:	Ф
dense_213_4115435:#
dense_214_4115438:
dense_214_4115440:#
dense_215_4115443:
dense_215_4115445:-
prediction_output_0_4115448:)
prediction_output_0_4115450:
identityИв.batch_normalization_71/StatefulPartitionedCallв"conv2d_426/StatefulPartitionedCallв"conv2d_427/StatefulPartitionedCallв"conv2d_428/StatefulPartitionedCallв"conv2d_429/StatefulPartitionedCallв"conv2d_430/StatefulPartitionedCallв"conv2d_431/StatefulPartitionedCallв!dense_213/StatefulPartitionedCallв!dense_214/StatefulPartitionedCallв!dense_215/StatefulPartitionedCallв+prediction_output_0/StatefulPartitionedCallв,spatial_dropout2d_71/StatefulPartitionedCallК
"conv2d_429/StatefulPartitionedCallStatefulPartitionedCall	input_144conv2d_429_4115389conv2d_429_4115391*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_429_layer_call_and_return_conditional_losses_4114720К
"conv2d_426/StatefulPartitionedCallStatefulPartitionedCall	input_143conv2d_426_4115394conv2d_426_4115396*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_426_layer_call_and_return_conditional_losses_4114737Т
,spatial_dropout2d_71/StatefulPartitionedCallStatefulPartitionedCall+conv2d_429/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_spatial_dropout2d_71_layer_call_and_return_conditional_losses_4114620Ю
.batch_normalization_71/StatefulPartitionedCallStatefulPartitionedCall+conv2d_426/StatefulPartitionedCall:output:0batch_normalization_71_4115400batch_normalization_71_4115402batch_normalization_71_4115404batch_normalization_71_4115406*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_71_layer_call_and_return_conditional_losses_4114676╢
"conv2d_430/StatefulPartitionedCallStatefulPartitionedCall5spatial_dropout2d_71/StatefulPartitionedCall:output:0conv2d_430_4115409conv2d_430_4115411*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_430_layer_call_and_return_conditional_losses_4114764╕
"conv2d_427/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_71/StatefulPartitionedCall:output:0conv2d_427_4115414conv2d_427_4115416*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_427_layer_call_and_return_conditional_losses_4114781м
"conv2d_431/StatefulPartitionedCallStatefulPartitionedCall+conv2d_430/StatefulPartitionedCall:output:0conv2d_431_4115419conv2d_431_4115421*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_431_layer_call_and_return_conditional_losses_4114798м
"conv2d_428/StatefulPartitionedCallStatefulPartitionedCall+conv2d_427/StatefulPartitionedCall:output:0conv2d_428_4115424conv2d_428_4115426*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_428_layer_call_and_return_conditional_losses_4114815 
'global_max_pooling2d_71/PartitionedCallPartitionedCall+conv2d_431/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *]
fXRV
T__inference_global_max_pooling2d_71_layer_call_and_return_conditional_losses_4114697ь
flatten_143/PartitionedCallPartitionedCall0global_max_pooling2d_71/PartitionedCall:output:0*
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
GPU2*0J 8В *Q
fLRJ
H__inference_flatten_143_layer_call_and_return_conditional_losses_4114828ш
flatten_142/PartitionedCallPartitionedCall+conv2d_428/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Т* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_flatten_142_layer_call_and_return_conditional_losses_4114836О
concatenate_71/PartitionedCallPartitionedCall$flatten_143/PartitionedCall:output:0$flatten_142/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ф* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_concatenate_71_layer_call_and_return_conditional_losses_4114845Ы
!dense_213/StatefulPartitionedCallStatefulPartitionedCall'concatenate_71/PartitionedCall:output:0dense_213_4115433dense_213_4115435*
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
GPU2*0J 8В *O
fJRH
F__inference_dense_213_layer_call_and_return_conditional_losses_4114858Ю
!dense_214/StatefulPartitionedCallStatefulPartitionedCall*dense_213/StatefulPartitionedCall:output:0dense_214_4115438dense_214_4115440*
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
GPU2*0J 8В *O
fJRH
F__inference_dense_214_layer_call_and_return_conditional_losses_4114875Ю
!dense_215/StatefulPartitionedCallStatefulPartitionedCall*dense_214/StatefulPartitionedCall:output:0dense_215_4115443dense_215_4115445*
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
GPU2*0J 8В *O
fJRH
F__inference_dense_215_layer_call_and_return_conditional_losses_4114892╞
+prediction_output_0/StatefulPartitionedCallStatefulPartitionedCall*dense_215/StatefulPartitionedCall:output:0prediction_output_0_4115448prediction_output_0_4115450*
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
GPU2*0J 8В *Y
fTRR
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4114908Г
IdentityIdentity4prediction_output_0/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Ю
NoOpNoOp/^batch_normalization_71/StatefulPartitionedCall#^conv2d_426/StatefulPartitionedCall#^conv2d_427/StatefulPartitionedCall#^conv2d_428/StatefulPartitionedCall#^conv2d_429/StatefulPartitionedCall#^conv2d_430/StatefulPartitionedCall#^conv2d_431/StatefulPartitionedCall"^dense_213/StatefulPartitionedCall"^dense_214/StatefulPartitionedCall"^dense_215/StatefulPartitionedCall,^prediction_output_0/StatefulPartitionedCall-^spatial_dropout2d_71/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:         Й:         ЙЙ: : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_71/StatefulPartitionedCall.batch_normalization_71/StatefulPartitionedCall2H
"conv2d_426/StatefulPartitionedCall"conv2d_426/StatefulPartitionedCall2H
"conv2d_427/StatefulPartitionedCall"conv2d_427/StatefulPartitionedCall2H
"conv2d_428/StatefulPartitionedCall"conv2d_428/StatefulPartitionedCall2H
"conv2d_429/StatefulPartitionedCall"conv2d_429/StatefulPartitionedCall2H
"conv2d_430/StatefulPartitionedCall"conv2d_430/StatefulPartitionedCall2H
"conv2d_431/StatefulPartitionedCall"conv2d_431/StatefulPartitionedCall2F
!dense_213/StatefulPartitionedCall!dense_213/StatefulPartitionedCall2F
!dense_214/StatefulPartitionedCall!dense_214/StatefulPartitionedCall2F
!dense_215/StatefulPartitionedCall!dense_215/StatefulPartitionedCall2Z
+prediction_output_0/StatefulPartitionedCall+prediction_output_0/StatefulPartitionedCall2\
,spatial_dropout2d_71/StatefulPartitionedCall,spatial_dropout2d_71/StatefulPartitionedCall:[ W
0
_output_shapes
:         Й
#
_user_specified_name	input_143:\X
1
_output_shapes
:         ЙЙ
#
_user_specified_name	input_144
▌
в
5__inference_prediction_output_0_layer_call_fn_4116171

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCallш
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
GPU2*0J 8В *Y
fTRR
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4114908o
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
╚
w
K__inference_concatenate_71_layer_call_and_return_conditional_losses_4116102
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :x
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:         ФX
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:         Ф"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         :         Т:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:         Т
"
_user_specified_name
inputs/1
б

°
F__inference_dense_213_layer_call_and_return_conditional_losses_4114858

inputs1
matmul_readvariableop_resource:	Ф-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ф*
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
_construction_contextkEagerRuntime*+
_input_shapes
:         Ф: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         Ф
 
_user_specified_nameinputs
ў
б
,__inference_conv2d_428_layer_call_fn_4116056

inputs!
unknown:
	unknown_0:
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_428_layer_call_and_return_conditional_losses_4114815x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         Й`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Й: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         Й
 
_user_specified_nameinputs
П
А
G__inference_conv2d_429_layer_call_and_return_conditional_losses_4115856

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0Ы
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides	
Йr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ЙY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Йj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Йw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЙЙ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЙЙ
 
_user_specified_nameinputs
М
А
G__inference_conv2d_428_layer_call_and_return_conditional_losses_4114815

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ЙY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Йj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Йw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Й: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         Й
 
_user_specified_nameinputs
╪N
ч
H__inference_joint_model_layer_call_and_return_conditional_losses_4115385
	input_143
	input_144,
conv2d_429_4115320:		 
conv2d_429_4115322:,
conv2d_426_4115325: 
conv2d_426_4115327:,
batch_normalization_71_4115331:,
batch_normalization_71_4115333:,
batch_normalization_71_4115335:,
batch_normalization_71_4115337:,
conv2d_430_4115340: 
conv2d_430_4115342:,
conv2d_427_4115345:	 
conv2d_427_4115347:,
conv2d_431_4115350: 
conv2d_431_4115352:,
conv2d_428_4115355: 
conv2d_428_4115357:$
dense_213_4115364:	Ф
dense_213_4115366:#
dense_214_4115369:
dense_214_4115371:#
dense_215_4115374:
dense_215_4115376:-
prediction_output_0_4115379:)
prediction_output_0_4115381:
identityИв.batch_normalization_71/StatefulPartitionedCallв"conv2d_426/StatefulPartitionedCallв"conv2d_427/StatefulPartitionedCallв"conv2d_428/StatefulPartitionedCallв"conv2d_429/StatefulPartitionedCallв"conv2d_430/StatefulPartitionedCallв"conv2d_431/StatefulPartitionedCallв!dense_213/StatefulPartitionedCallв!dense_214/StatefulPartitionedCallв!dense_215/StatefulPartitionedCallв+prediction_output_0/StatefulPartitionedCallК
"conv2d_429/StatefulPartitionedCallStatefulPartitionedCall	input_144conv2d_429_4115320conv2d_429_4115322*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_429_layer_call_and_return_conditional_losses_4114720К
"conv2d_426/StatefulPartitionedCallStatefulPartitionedCall	input_143conv2d_426_4115325conv2d_426_4115327*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_426_layer_call_and_return_conditional_losses_4114737В
$spatial_dropout2d_71/PartitionedCallPartitionedCall+conv2d_429/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_spatial_dropout2d_71_layer_call_and_return_conditional_losses_4114592а
.batch_normalization_71/StatefulPartitionedCallStatefulPartitionedCall+conv2d_426/StatefulPartitionedCall:output:0batch_normalization_71_4115331batch_normalization_71_4115333batch_normalization_71_4115335batch_normalization_71_4115337*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_71_layer_call_and_return_conditional_losses_4114645о
"conv2d_430/StatefulPartitionedCallStatefulPartitionedCall-spatial_dropout2d_71/PartitionedCall:output:0conv2d_430_4115340conv2d_430_4115342*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_430_layer_call_and_return_conditional_losses_4114764╕
"conv2d_427/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_71/StatefulPartitionedCall:output:0conv2d_427_4115345conv2d_427_4115347*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_427_layer_call_and_return_conditional_losses_4114781м
"conv2d_431/StatefulPartitionedCallStatefulPartitionedCall+conv2d_430/StatefulPartitionedCall:output:0conv2d_431_4115350conv2d_431_4115352*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_431_layer_call_and_return_conditional_losses_4114798м
"conv2d_428/StatefulPartitionedCallStatefulPartitionedCall+conv2d_427/StatefulPartitionedCall:output:0conv2d_428_4115355conv2d_428_4115357*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_428_layer_call_and_return_conditional_losses_4114815 
'global_max_pooling2d_71/PartitionedCallPartitionedCall+conv2d_431/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *]
fXRV
T__inference_global_max_pooling2d_71_layer_call_and_return_conditional_losses_4114697ь
flatten_143/PartitionedCallPartitionedCall0global_max_pooling2d_71/PartitionedCall:output:0*
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
GPU2*0J 8В *Q
fLRJ
H__inference_flatten_143_layer_call_and_return_conditional_losses_4114828ш
flatten_142/PartitionedCallPartitionedCall+conv2d_428/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Т* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_flatten_142_layer_call_and_return_conditional_losses_4114836О
concatenate_71/PartitionedCallPartitionedCall$flatten_143/PartitionedCall:output:0$flatten_142/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ф* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_concatenate_71_layer_call_and_return_conditional_losses_4114845Ы
!dense_213/StatefulPartitionedCallStatefulPartitionedCall'concatenate_71/PartitionedCall:output:0dense_213_4115364dense_213_4115366*
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
GPU2*0J 8В *O
fJRH
F__inference_dense_213_layer_call_and_return_conditional_losses_4114858Ю
!dense_214/StatefulPartitionedCallStatefulPartitionedCall*dense_213/StatefulPartitionedCall:output:0dense_214_4115369dense_214_4115371*
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
GPU2*0J 8В *O
fJRH
F__inference_dense_214_layer_call_and_return_conditional_losses_4114875Ю
!dense_215/StatefulPartitionedCallStatefulPartitionedCall*dense_214/StatefulPartitionedCall:output:0dense_215_4115374dense_215_4115376*
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
GPU2*0J 8В *O
fJRH
F__inference_dense_215_layer_call_and_return_conditional_losses_4114892╞
+prediction_output_0/StatefulPartitionedCallStatefulPartitionedCall*dense_215/StatefulPartitionedCall:output:0prediction_output_0_4115379prediction_output_0_4115381*
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
GPU2*0J 8В *Y
fTRR
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4114908Г
IdentityIdentity4prediction_output_0/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         я
NoOpNoOp/^batch_normalization_71/StatefulPartitionedCall#^conv2d_426/StatefulPartitionedCall#^conv2d_427/StatefulPartitionedCall#^conv2d_428/StatefulPartitionedCall#^conv2d_429/StatefulPartitionedCall#^conv2d_430/StatefulPartitionedCall#^conv2d_431/StatefulPartitionedCall"^dense_213/StatefulPartitionedCall"^dense_214/StatefulPartitionedCall"^dense_215/StatefulPartitionedCall,^prediction_output_0/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:         Й:         ЙЙ: : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_71/StatefulPartitionedCall.batch_normalization_71/StatefulPartitionedCall2H
"conv2d_426/StatefulPartitionedCall"conv2d_426/StatefulPartitionedCall2H
"conv2d_427/StatefulPartitionedCall"conv2d_427/StatefulPartitionedCall2H
"conv2d_428/StatefulPartitionedCall"conv2d_428/StatefulPartitionedCall2H
"conv2d_429/StatefulPartitionedCall"conv2d_429/StatefulPartitionedCall2H
"conv2d_430/StatefulPartitionedCall"conv2d_430/StatefulPartitionedCall2H
"conv2d_431/StatefulPartitionedCall"conv2d_431/StatefulPartitionedCall2F
!dense_213/StatefulPartitionedCall!dense_213/StatefulPartitionedCall2F
!dense_214/StatefulPartitionedCall!dense_214/StatefulPartitionedCall2F
!dense_215/StatefulPartitionedCall!dense_215/StatefulPartitionedCall2Z
+prediction_output_0/StatefulPartitionedCall+prediction_output_0/StatefulPartitionedCall:[ W
0
_output_shapes
:         Й
#
_user_specified_name	input_143:\X
1
_output_shapes
:         ЙЙ
#
_user_specified_name	input_144
Ш	
╙
8__inference_batch_normalization_71_layer_call_fn_4115947

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallЯ
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
GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_71_layer_call_and_return_conditional_losses_4114645Й
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
Мо
╒0
#__inference__traced_restore_4116665
file_prefix<
"assignvariableop_conv2d_429_kernel:		0
"assignvariableop_1_conv2d_429_bias:>
$assignvariableop_2_conv2d_426_kernel:0
"assignvariableop_3_conv2d_426_bias:>
$assignvariableop_4_conv2d_430_kernel:0
"assignvariableop_5_conv2d_430_bias:=
/assignvariableop_6_batch_normalization_71_gamma:<
.assignvariableop_7_batch_normalization_71_beta:C
5assignvariableop_8_batch_normalization_71_moving_mean:G
9assignvariableop_9_batch_normalization_71_moving_variance:?
%assignvariableop_10_conv2d_431_kernel:1
#assignvariableop_11_conv2d_431_bias:?
%assignvariableop_12_conv2d_427_kernel:	1
#assignvariableop_13_conv2d_427_bias:?
%assignvariableop_14_conv2d_428_kernel:1
#assignvariableop_15_conv2d_428_bias:7
$assignvariableop_16_dense_213_kernel:	Ф0
"assignvariableop_17_dense_213_bias:6
$assignvariableop_18_dense_214_kernel:0
"assignvariableop_19_dense_214_bias:6
$assignvariableop_20_dense_215_kernel:0
"assignvariableop_21_dense_215_bias:@
.assignvariableop_22_prediction_output_0_kernel::
,assignvariableop_23_prediction_output_0_bias:$
assignvariableop_24_beta_1: $
assignvariableop_25_beta_2: #
assignvariableop_26_decay: +
!assignvariableop_27_learning_rate: '
assignvariableop_28_adam_iter:	 #
assignvariableop_29_total: #
assignvariableop_30_count: F
,assignvariableop_31_adam_conv2d_429_kernel_m:		8
*assignvariableop_32_adam_conv2d_429_bias_m:F
,assignvariableop_33_adam_conv2d_426_kernel_m:8
*assignvariableop_34_adam_conv2d_426_bias_m:F
,assignvariableop_35_adam_conv2d_430_kernel_m:8
*assignvariableop_36_adam_conv2d_430_bias_m:E
7assignvariableop_37_adam_batch_normalization_71_gamma_m:D
6assignvariableop_38_adam_batch_normalization_71_beta_m:F
,assignvariableop_39_adam_conv2d_431_kernel_m:8
*assignvariableop_40_adam_conv2d_431_bias_m:F
,assignvariableop_41_adam_conv2d_427_kernel_m:	8
*assignvariableop_42_adam_conv2d_427_bias_m:F
,assignvariableop_43_adam_conv2d_428_kernel_m:8
*assignvariableop_44_adam_conv2d_428_bias_m:>
+assignvariableop_45_adam_dense_213_kernel_m:	Ф7
)assignvariableop_46_adam_dense_213_bias_m:=
+assignvariableop_47_adam_dense_214_kernel_m:7
)assignvariableop_48_adam_dense_214_bias_m:=
+assignvariableop_49_adam_dense_215_kernel_m:7
)assignvariableop_50_adam_dense_215_bias_m:G
5assignvariableop_51_adam_prediction_output_0_kernel_m:A
3assignvariableop_52_adam_prediction_output_0_bias_m:F
,assignvariableop_53_adam_conv2d_429_kernel_v:		8
*assignvariableop_54_adam_conv2d_429_bias_v:F
,assignvariableop_55_adam_conv2d_426_kernel_v:8
*assignvariableop_56_adam_conv2d_426_bias_v:F
,assignvariableop_57_adam_conv2d_430_kernel_v:8
*assignvariableop_58_adam_conv2d_430_bias_v:E
7assignvariableop_59_adam_batch_normalization_71_gamma_v:D
6assignvariableop_60_adam_batch_normalization_71_beta_v:F
,assignvariableop_61_adam_conv2d_431_kernel_v:8
*assignvariableop_62_adam_conv2d_431_bias_v:F
,assignvariableop_63_adam_conv2d_427_kernel_v:	8
*assignvariableop_64_adam_conv2d_427_bias_v:F
,assignvariableop_65_adam_conv2d_428_kernel_v:8
*assignvariableop_66_adam_conv2d_428_bias_v:>
+assignvariableop_67_adam_dense_213_kernel_v:	Ф7
)assignvariableop_68_adam_dense_213_bias_v:=
+assignvariableop_69_adam_dense_214_kernel_v:7
)assignvariableop_70_adam_dense_214_bias_v:=
+assignvariableop_71_adam_dense_215_kernel_v:7
)assignvariableop_72_adam_dense_215_bias_v:G
5assignvariableop_73_adam_prediction_output_0_kernel_v:A
3assignvariableop_74_adam_prediction_output_0_bias_v:
identity_76ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_56вAssignVariableOp_57вAssignVariableOp_58вAssignVariableOp_59вAssignVariableOp_6вAssignVariableOp_60вAssignVariableOp_61вAssignVariableOp_62вAssignVariableOp_63вAssignVariableOp_64вAssignVariableOp_65вAssignVariableOp_66вAssignVariableOp_67вAssignVariableOp_68вAssignVariableOp_69вAssignVariableOp_7вAssignVariableOp_70вAssignVariableOp_71вAssignVariableOp_72вAssignVariableOp_73вAssignVariableOp_74вAssignVariableOp_8вAssignVariableOp_9▌*
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*Г*
value∙)BЎ)LB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЛ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*н
valueгBаLB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Э
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*╞
_output_shapes│
░::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Z
dtypesP
N2L	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_429_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_429_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_2AssignVariableOp$assignvariableop_2_conv2d_426_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_3AssignVariableOp"assignvariableop_3_conv2d_426_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_4AssignVariableOp$assignvariableop_4_conv2d_430_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_5AssignVariableOp"assignvariableop_5_conv2d_430_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_71_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_71_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_8AssignVariableOp5assignvariableop_8_batch_normalization_71_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_9AssignVariableOp9assignvariableop_9_batch_normalization_71_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_10AssignVariableOp%assignvariableop_10_conv2d_431_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_11AssignVariableOp#assignvariableop_11_conv2d_431_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_12AssignVariableOp%assignvariableop_12_conv2d_427_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_13AssignVariableOp#assignvariableop_13_conv2d_427_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_14AssignVariableOp%assignvariableop_14_conv2d_428_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_15AssignVariableOp#assignvariableop_15_conv2d_428_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_16AssignVariableOp$assignvariableop_16_dense_213_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_17AssignVariableOp"assignvariableop_17_dense_213_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_18AssignVariableOp$assignvariableop_18_dense_214_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_19AssignVariableOp"assignvariableop_19_dense_214_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_20AssignVariableOp$assignvariableop_20_dense_215_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_21AssignVariableOp"assignvariableop_21_dense_215_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_22AssignVariableOp.assignvariableop_22_prediction_output_0_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_23AssignVariableOp,assignvariableop_23_prediction_output_0_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_24AssignVariableOpassignvariableop_24_beta_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_25AssignVariableOpassignvariableop_25_beta_2Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_26AssignVariableOpassignvariableop_26_decayIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_27AssignVariableOp!assignvariableop_27_learning_rateIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0	*
_output_shapes
:О
AssignVariableOp_28AssignVariableOpassignvariableop_28_adam_iterIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_29AssignVariableOpassignvariableop_29_totalIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_30AssignVariableOpassignvariableop_30_countIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_31AssignVariableOp,assignvariableop_31_adam_conv2d_429_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_conv2d_429_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_33AssignVariableOp,assignvariableop_33_adam_conv2d_426_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_conv2d_426_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_35AssignVariableOp,assignvariableop_35_adam_conv2d_430_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_36AssignVariableOp*assignvariableop_36_adam_conv2d_430_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_37AssignVariableOp7assignvariableop_37_adam_batch_normalization_71_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_38AssignVariableOp6assignvariableop_38_adam_batch_normalization_71_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_39AssignVariableOp,assignvariableop_39_adam_conv2d_431_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_conv2d_431_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_41AssignVariableOp,assignvariableop_41_adam_conv2d_427_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_conv2d_427_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_43AssignVariableOp,assignvariableop_43_adam_conv2d_428_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_conv2d_428_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_dense_213_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_dense_213_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_dense_214_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_dense_214_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_dense_215_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_dense_215_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_51AssignVariableOp5assignvariableop_51_adam_prediction_output_0_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_52AssignVariableOp3assignvariableop_52_adam_prediction_output_0_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_53AssignVariableOp,assignvariableop_53_adam_conv2d_429_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_54AssignVariableOp*assignvariableop_54_adam_conv2d_429_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_55AssignVariableOp,assignvariableop_55_adam_conv2d_426_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_56AssignVariableOp*assignvariableop_56_adam_conv2d_426_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_57AssignVariableOp,assignvariableop_57_adam_conv2d_430_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_conv2d_430_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_59AssignVariableOp7assignvariableop_59_adam_batch_normalization_71_gamma_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_60AssignVariableOp6assignvariableop_60_adam_batch_normalization_71_beta_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_61AssignVariableOp,assignvariableop_61_adam_conv2d_431_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_62AssignVariableOp*assignvariableop_62_adam_conv2d_431_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_63AssignVariableOp,assignvariableop_63_adam_conv2d_427_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_64AssignVariableOp*assignvariableop_64_adam_conv2d_427_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_65AssignVariableOp,assignvariableop_65_adam_conv2d_428_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_66AssignVariableOp*assignvariableop_66_adam_conv2d_428_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_dense_213_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_dense_213_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_dense_214_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_dense_214_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_dense_215_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_dense_215_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_73AssignVariableOp5assignvariableop_73_adam_prediction_output_0_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_74AssignVariableOp3assignvariableop_74_adam_prediction_output_0_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ┴
Identity_75Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_76IdentityIdentity_75:output:0^NoOp_1*
T0*
_output_shapes
: о
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_76Identity_76:output:0*н
_input_shapesЫ
Ш: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_74AssignVariableOp_742(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
░
p
T__inference_global_max_pooling2d_71_layer_call_and_return_conditional_losses_4116047

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
╬
Ю
S__inference_batch_normalization_71_layer_call_and_return_conditional_losses_4115978

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
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
╖
\
0__inference_concatenate_71_layer_call_fn_4116095
inputs_0
inputs_1
identity╟
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ф* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_concatenate_71_layer_call_and_return_conditional_losses_4114845a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         Ф"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         :         Т:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:         Т
"
_user_specified_name
inputs/1
╙	
Б
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4114908

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
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
М
А
G__inference_conv2d_427_layer_call_and_return_conditional_losses_4116036

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ЙY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Йj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Йw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Й: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         Й
 
_user_specified_nameinputs
Ё
o
Q__inference_spatial_dropout2d_71_layer_call_and_return_conditional_losses_4114592

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
М
А
G__inference_conv2d_427_layer_call_and_return_conditional_losses_4114781

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ЙY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Йj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Йw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Й: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         Й
 
_user_specified_nameinputs
ЬP
Т
H__inference_joint_model_layer_call_and_return_conditional_losses_4115211

inputs
inputs_1,
conv2d_429_4115146:		 
conv2d_429_4115148:,
conv2d_426_4115151: 
conv2d_426_4115153:,
batch_normalization_71_4115157:,
batch_normalization_71_4115159:,
batch_normalization_71_4115161:,
batch_normalization_71_4115163:,
conv2d_430_4115166: 
conv2d_430_4115168:,
conv2d_427_4115171:	 
conv2d_427_4115173:,
conv2d_431_4115176: 
conv2d_431_4115178:,
conv2d_428_4115181: 
conv2d_428_4115183:$
dense_213_4115190:	Ф
dense_213_4115192:#
dense_214_4115195:
dense_214_4115197:#
dense_215_4115200:
dense_215_4115202:-
prediction_output_0_4115205:)
prediction_output_0_4115207:
identityИв.batch_normalization_71/StatefulPartitionedCallв"conv2d_426/StatefulPartitionedCallв"conv2d_427/StatefulPartitionedCallв"conv2d_428/StatefulPartitionedCallв"conv2d_429/StatefulPartitionedCallв"conv2d_430/StatefulPartitionedCallв"conv2d_431/StatefulPartitionedCallв!dense_213/StatefulPartitionedCallв!dense_214/StatefulPartitionedCallв!dense_215/StatefulPartitionedCallв+prediction_output_0/StatefulPartitionedCallв,spatial_dropout2d_71/StatefulPartitionedCallЙ
"conv2d_429/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv2d_429_4115146conv2d_429_4115148*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_429_layer_call_and_return_conditional_losses_4114720З
"conv2d_426/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_426_4115151conv2d_426_4115153*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_426_layer_call_and_return_conditional_losses_4114737Т
,spatial_dropout2d_71/StatefulPartitionedCallStatefulPartitionedCall+conv2d_429/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_spatial_dropout2d_71_layer_call_and_return_conditional_losses_4114620Ю
.batch_normalization_71/StatefulPartitionedCallStatefulPartitionedCall+conv2d_426/StatefulPartitionedCall:output:0batch_normalization_71_4115157batch_normalization_71_4115159batch_normalization_71_4115161batch_normalization_71_4115163*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_71_layer_call_and_return_conditional_losses_4114676╢
"conv2d_430/StatefulPartitionedCallStatefulPartitionedCall5spatial_dropout2d_71/StatefulPartitionedCall:output:0conv2d_430_4115166conv2d_430_4115168*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_430_layer_call_and_return_conditional_losses_4114764╕
"conv2d_427/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_71/StatefulPartitionedCall:output:0conv2d_427_4115171conv2d_427_4115173*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_427_layer_call_and_return_conditional_losses_4114781м
"conv2d_431/StatefulPartitionedCallStatefulPartitionedCall+conv2d_430/StatefulPartitionedCall:output:0conv2d_431_4115176conv2d_431_4115178*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_431_layer_call_and_return_conditional_losses_4114798м
"conv2d_428/StatefulPartitionedCallStatefulPartitionedCall+conv2d_427/StatefulPartitionedCall:output:0conv2d_428_4115181conv2d_428_4115183*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_428_layer_call_and_return_conditional_losses_4114815 
'global_max_pooling2d_71/PartitionedCallPartitionedCall+conv2d_431/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *]
fXRV
T__inference_global_max_pooling2d_71_layer_call_and_return_conditional_losses_4114697ь
flatten_143/PartitionedCallPartitionedCall0global_max_pooling2d_71/PartitionedCall:output:0*
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
GPU2*0J 8В *Q
fLRJ
H__inference_flatten_143_layer_call_and_return_conditional_losses_4114828ш
flatten_142/PartitionedCallPartitionedCall+conv2d_428/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Т* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_flatten_142_layer_call_and_return_conditional_losses_4114836О
concatenate_71/PartitionedCallPartitionedCall$flatten_143/PartitionedCall:output:0$flatten_142/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ф* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_concatenate_71_layer_call_and_return_conditional_losses_4114845Ы
!dense_213/StatefulPartitionedCallStatefulPartitionedCall'concatenate_71/PartitionedCall:output:0dense_213_4115190dense_213_4115192*
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
GPU2*0J 8В *O
fJRH
F__inference_dense_213_layer_call_and_return_conditional_losses_4114858Ю
!dense_214/StatefulPartitionedCallStatefulPartitionedCall*dense_213/StatefulPartitionedCall:output:0dense_214_4115195dense_214_4115197*
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
GPU2*0J 8В *O
fJRH
F__inference_dense_214_layer_call_and_return_conditional_losses_4114875Ю
!dense_215/StatefulPartitionedCallStatefulPartitionedCall*dense_214/StatefulPartitionedCall:output:0dense_215_4115200dense_215_4115202*
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
GPU2*0J 8В *O
fJRH
F__inference_dense_215_layer_call_and_return_conditional_losses_4114892╞
+prediction_output_0/StatefulPartitionedCallStatefulPartitionedCall*dense_215/StatefulPartitionedCall:output:0prediction_output_0_4115205prediction_output_0_4115207*
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
GPU2*0J 8В *Y
fTRR
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4114908Г
IdentityIdentity4prediction_output_0/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Ю
NoOpNoOp/^batch_normalization_71/StatefulPartitionedCall#^conv2d_426/StatefulPartitionedCall#^conv2d_427/StatefulPartitionedCall#^conv2d_428/StatefulPartitionedCall#^conv2d_429/StatefulPartitionedCall#^conv2d_430/StatefulPartitionedCall#^conv2d_431/StatefulPartitionedCall"^dense_213/StatefulPartitionedCall"^dense_214/StatefulPartitionedCall"^dense_215/StatefulPartitionedCall,^prediction_output_0/StatefulPartitionedCall-^spatial_dropout2d_71/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:         Й:         ЙЙ: : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_71/StatefulPartitionedCall.batch_normalization_71/StatefulPartitionedCall2H
"conv2d_426/StatefulPartitionedCall"conv2d_426/StatefulPartitionedCall2H
"conv2d_427/StatefulPartitionedCall"conv2d_427/StatefulPartitionedCall2H
"conv2d_428/StatefulPartitionedCall"conv2d_428/StatefulPartitionedCall2H
"conv2d_429/StatefulPartitionedCall"conv2d_429/StatefulPartitionedCall2H
"conv2d_430/StatefulPartitionedCall"conv2d_430/StatefulPartitionedCall2H
"conv2d_431/StatefulPartitionedCall"conv2d_431/StatefulPartitionedCall2F
!dense_213/StatefulPartitionedCall!dense_213/StatefulPartitionedCall2F
!dense_214/StatefulPartitionedCall!dense_214/StatefulPartitionedCall2F
!dense_215/StatefulPartitionedCall!dense_215/StatefulPartitionedCall2Z
+prediction_output_0/StatefulPartitionedCall+prediction_output_0/StatefulPartitionedCall2\
,spatial_dropout2d_71/StatefulPartitionedCall,spatial_dropout2d_71/StatefulPartitionedCall:X T
0
_output_shapes
:         Й
 
_user_specified_nameinputs:YU
1
_output_shapes
:         ЙЙ
 
_user_specified_nameinputs
А
p
Q__inference_spatial_dropout2d_71_layer_call_and_return_conditional_losses_4115894

inputs
identityИ;
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
valueB:╤
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
 *  а?З
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
value	B :╓
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:м
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
 *═╠L>╖
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"                  А
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"                  М
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
╕
d
H__inference_flatten_143_layer_call_and_return_conditional_losses_4114828

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
Э

ў
F__inference_dense_215_layer_call_and_return_conditional_losses_4114892

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
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
Ё
o
Q__inference_spatial_dropout2d_71_layer_call_and_return_conditional_losses_4115871

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
М
А
G__inference_conv2d_428_layer_call_and_return_conditional_losses_4116067

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ЙY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Йj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Йw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Й: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         Й
 
_user_specified_nameinputs
ў
б
,__inference_conv2d_427_layer_call_fn_4116025

inputs!
unknown:	
	unknown_0:
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_427_layer_call_and_return_conditional_losses_4114781x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         Й`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Й: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         Й
 
_user_specified_nameinputs
ў
б
,__inference_conv2d_431_layer_call_fn_4116005

inputs!
unknown:
	unknown_0:
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_431_layer_call_and_return_conditional_losses_4114798x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         Й`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Й: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         Й
 
_user_specified_nameinputs
Э

ў
F__inference_dense_214_layer_call_and_return_conditional_losses_4114875

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
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
╠
Щ
+__inference_dense_213_layer_call_fn_4116111

inputs
unknown:	Ф
	unknown_0:
identityИвStatefulPartitionedCall▐
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
GPU2*0J 8В *O
fJRH
F__inference_dense_213_layer_call_and_return_conditional_losses_4114858o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ф: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Ф
 
_user_specified_nameinputs
И
┬
S__inference_batch_normalization_71_layer_call_and_return_conditional_losses_4114676

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ╘
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
∙
б
,__inference_conv2d_429_layer_call_fn_4115845

inputs!
unknown:		
	unknown_0:
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_429_layer_call_and_return_conditional_losses_4114720x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         Й`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЙЙ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЙЙ
 
_user_specified_nameinputs
Щ
o
6__inference_spatial_dropout2d_71_layer_call_fn_4115866

inputs
identityИвStatefulPartitionedCallЄ
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
GPU2*0J 8В *Z
fURS
Q__inference_spatial_dropout2d_71_layer_call_and_return_conditional_losses_4114620Т
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
╟
R
6__inference_spatial_dropout2d_71_layer_call_fn_4115861

inputs
identityт
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
GPU2*0J 8В *Z
fURS
Q__inference_spatial_dropout2d_71_layer_call_and_return_conditional_losses_4114592Г
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
П
А
G__inference_conv2d_429_layer_call_and_return_conditional_losses_4114720

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0Ы
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides	
Йr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ЙY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Йj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Йw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЙЙ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЙЙ
 
_user_specified_nameinputs
▌
┤
-__inference_joint_model_layer_call_fn_4114966
	input_143
	input_144!
unknown:		
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:	

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:	Ф

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identityИвStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCall	input_143	input_144unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_joint_model_layer_call_and_return_conditional_losses_4114915o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:         Й:         ЙЙ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
0
_output_shapes
:         Й
#
_user_specified_name	input_143:\X
1
_output_shapes
:         ЙЙ
#
_user_specified_name	input_144
╬
Ю
S__inference_batch_normalization_71_layer_call_and_return_conditional_losses_4114645

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
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
ўФ
╬ 
 __inference__traced_save_4116430
file_prefix0
,savev2_conv2d_429_kernel_read_readvariableop.
*savev2_conv2d_429_bias_read_readvariableop0
,savev2_conv2d_426_kernel_read_readvariableop.
*savev2_conv2d_426_bias_read_readvariableop0
,savev2_conv2d_430_kernel_read_readvariableop.
*savev2_conv2d_430_bias_read_readvariableop;
7savev2_batch_normalization_71_gamma_read_readvariableop:
6savev2_batch_normalization_71_beta_read_readvariableopA
=savev2_batch_normalization_71_moving_mean_read_readvariableopE
Asavev2_batch_normalization_71_moving_variance_read_readvariableop0
,savev2_conv2d_431_kernel_read_readvariableop.
*savev2_conv2d_431_bias_read_readvariableop0
,savev2_conv2d_427_kernel_read_readvariableop.
*savev2_conv2d_427_bias_read_readvariableop0
,savev2_conv2d_428_kernel_read_readvariableop.
*savev2_conv2d_428_bias_read_readvariableop/
+savev2_dense_213_kernel_read_readvariableop-
)savev2_dense_213_bias_read_readvariableop/
+savev2_dense_214_kernel_read_readvariableop-
)savev2_dense_214_bias_read_readvariableop/
+savev2_dense_215_kernel_read_readvariableop-
)savev2_dense_215_bias_read_readvariableop9
5savev2_prediction_output_0_kernel_read_readvariableop7
3savev2_prediction_output_0_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop7
3savev2_adam_conv2d_429_kernel_m_read_readvariableop5
1savev2_adam_conv2d_429_bias_m_read_readvariableop7
3savev2_adam_conv2d_426_kernel_m_read_readvariableop5
1savev2_adam_conv2d_426_bias_m_read_readvariableop7
3savev2_adam_conv2d_430_kernel_m_read_readvariableop5
1savev2_adam_conv2d_430_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_71_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_71_beta_m_read_readvariableop7
3savev2_adam_conv2d_431_kernel_m_read_readvariableop5
1savev2_adam_conv2d_431_bias_m_read_readvariableop7
3savev2_adam_conv2d_427_kernel_m_read_readvariableop5
1savev2_adam_conv2d_427_bias_m_read_readvariableop7
3savev2_adam_conv2d_428_kernel_m_read_readvariableop5
1savev2_adam_conv2d_428_bias_m_read_readvariableop6
2savev2_adam_dense_213_kernel_m_read_readvariableop4
0savev2_adam_dense_213_bias_m_read_readvariableop6
2savev2_adam_dense_214_kernel_m_read_readvariableop4
0savev2_adam_dense_214_bias_m_read_readvariableop6
2savev2_adam_dense_215_kernel_m_read_readvariableop4
0savev2_adam_dense_215_bias_m_read_readvariableop@
<savev2_adam_prediction_output_0_kernel_m_read_readvariableop>
:savev2_adam_prediction_output_0_bias_m_read_readvariableop7
3savev2_adam_conv2d_429_kernel_v_read_readvariableop5
1savev2_adam_conv2d_429_bias_v_read_readvariableop7
3savev2_adam_conv2d_426_kernel_v_read_readvariableop5
1savev2_adam_conv2d_426_bias_v_read_readvariableop7
3savev2_adam_conv2d_430_kernel_v_read_readvariableop5
1savev2_adam_conv2d_430_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_71_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_71_beta_v_read_readvariableop7
3savev2_adam_conv2d_431_kernel_v_read_readvariableop5
1savev2_adam_conv2d_431_bias_v_read_readvariableop7
3savev2_adam_conv2d_427_kernel_v_read_readvariableop5
1savev2_adam_conv2d_427_bias_v_read_readvariableop7
3savev2_adam_conv2d_428_kernel_v_read_readvariableop5
1savev2_adam_conv2d_428_bias_v_read_readvariableop6
2savev2_adam_dense_213_kernel_v_read_readvariableop4
0savev2_adam_dense_213_bias_v_read_readvariableop6
2savev2_adam_dense_214_kernel_v_read_readvariableop4
0savev2_adam_dense_214_bias_v_read_readvariableop6
2savev2_adam_dense_215_kernel_v_read_readvariableop4
0savev2_adam_dense_215_bias_v_read_readvariableop@
<savev2_adam_prediction_output_0_kernel_v_read_readvariableop>
:savev2_adam_prediction_output_0_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ┌*
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*Г*
value∙)BЎ)LB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHИ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*н
valueгBаLB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ░
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv2d_429_kernel_read_readvariableop*savev2_conv2d_429_bias_read_readvariableop,savev2_conv2d_426_kernel_read_readvariableop*savev2_conv2d_426_bias_read_readvariableop,savev2_conv2d_430_kernel_read_readvariableop*savev2_conv2d_430_bias_read_readvariableop7savev2_batch_normalization_71_gamma_read_readvariableop6savev2_batch_normalization_71_beta_read_readvariableop=savev2_batch_normalization_71_moving_mean_read_readvariableopAsavev2_batch_normalization_71_moving_variance_read_readvariableop,savev2_conv2d_431_kernel_read_readvariableop*savev2_conv2d_431_bias_read_readvariableop,savev2_conv2d_427_kernel_read_readvariableop*savev2_conv2d_427_bias_read_readvariableop,savev2_conv2d_428_kernel_read_readvariableop*savev2_conv2d_428_bias_read_readvariableop+savev2_dense_213_kernel_read_readvariableop)savev2_dense_213_bias_read_readvariableop+savev2_dense_214_kernel_read_readvariableop)savev2_dense_214_bias_read_readvariableop+savev2_dense_215_kernel_read_readvariableop)savev2_dense_215_bias_read_readvariableop5savev2_prediction_output_0_kernel_read_readvariableop3savev2_prediction_output_0_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop3savev2_adam_conv2d_429_kernel_m_read_readvariableop1savev2_adam_conv2d_429_bias_m_read_readvariableop3savev2_adam_conv2d_426_kernel_m_read_readvariableop1savev2_adam_conv2d_426_bias_m_read_readvariableop3savev2_adam_conv2d_430_kernel_m_read_readvariableop1savev2_adam_conv2d_430_bias_m_read_readvariableop>savev2_adam_batch_normalization_71_gamma_m_read_readvariableop=savev2_adam_batch_normalization_71_beta_m_read_readvariableop3savev2_adam_conv2d_431_kernel_m_read_readvariableop1savev2_adam_conv2d_431_bias_m_read_readvariableop3savev2_adam_conv2d_427_kernel_m_read_readvariableop1savev2_adam_conv2d_427_bias_m_read_readvariableop3savev2_adam_conv2d_428_kernel_m_read_readvariableop1savev2_adam_conv2d_428_bias_m_read_readvariableop2savev2_adam_dense_213_kernel_m_read_readvariableop0savev2_adam_dense_213_bias_m_read_readvariableop2savev2_adam_dense_214_kernel_m_read_readvariableop0savev2_adam_dense_214_bias_m_read_readvariableop2savev2_adam_dense_215_kernel_m_read_readvariableop0savev2_adam_dense_215_bias_m_read_readvariableop<savev2_adam_prediction_output_0_kernel_m_read_readvariableop:savev2_adam_prediction_output_0_bias_m_read_readvariableop3savev2_adam_conv2d_429_kernel_v_read_readvariableop1savev2_adam_conv2d_429_bias_v_read_readvariableop3savev2_adam_conv2d_426_kernel_v_read_readvariableop1savev2_adam_conv2d_426_bias_v_read_readvariableop3savev2_adam_conv2d_430_kernel_v_read_readvariableop1savev2_adam_conv2d_430_bias_v_read_readvariableop>savev2_adam_batch_normalization_71_gamma_v_read_readvariableop=savev2_adam_batch_normalization_71_beta_v_read_readvariableop3savev2_adam_conv2d_431_kernel_v_read_readvariableop1savev2_adam_conv2d_431_bias_v_read_readvariableop3savev2_adam_conv2d_427_kernel_v_read_readvariableop1savev2_adam_conv2d_427_bias_v_read_readvariableop3savev2_adam_conv2d_428_kernel_v_read_readvariableop1savev2_adam_conv2d_428_bias_v_read_readvariableop2savev2_adam_dense_213_kernel_v_read_readvariableop0savev2_adam_dense_213_bias_v_read_readvariableop2savev2_adam_dense_214_kernel_v_read_readvariableop0savev2_adam_dense_214_bias_v_read_readvariableop2savev2_adam_dense_215_kernel_v_read_readvariableop0savev2_adam_dense_215_bias_v_read_readvariableop<savev2_adam_prediction_output_0_kernel_v_read_readvariableop:savev2_adam_prediction_output_0_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Z
dtypesP
N2L	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
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

identity_1Identity_1:output:0*╩
_input_shapes╕
╡: :		::::::::::::	::::	Ф:::::::: : : : : : : :		::::::::::	::::	Ф::::::::		::::::::::	::::	Ф:::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:		: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 
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
:: 
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
::%!

_output_shapes
:	Ф: 
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
::

_output_shapes
: :

_output_shapes
: :
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
: :, (
&
_output_shapes
:		: !

_output_shapes
::,"(
&
_output_shapes
:: #

_output_shapes
::,$(
&
_output_shapes
:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
::,((
&
_output_shapes
:: )

_output_shapes
::,*(
&
_output_shapes
:	: +

_output_shapes
::,,(
&
_output_shapes
:: -

_output_shapes
::%.!

_output_shapes
:	Ф: /

_output_shapes
::$0 

_output_shapes

:: 1

_output_shapes
::$2 

_output_shapes

:: 3

_output_shapes
::$4 

_output_shapes

:: 5

_output_shapes
::,6(
&
_output_shapes
:		: 7

_output_shapes
::,8(
&
_output_shapes
:: 9

_output_shapes
::,:(
&
_output_shapes
:: ;

_output_shapes
:: <

_output_shapes
:: =

_output_shapes
::,>(
&
_output_shapes
:: ?

_output_shapes
::,@(
&
_output_shapes
:	: A

_output_shapes
::,B(
&
_output_shapes
:: C

_output_shapes
::%D!

_output_shapes
:	Ф: E

_output_shapes
::$F 

_output_shapes

:: G

_output_shapes
::$H 

_output_shapes

:: I

_output_shapes
::$J 

_output_shapes

:: K

_output_shapes
::L

_output_shapes
: 
Н
А
G__inference_conv2d_430_layer_call_and_return_conditional_losses_4114764

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ы
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides	
Йr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ЙY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Йj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Йw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Й: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         Й
 
_user_specified_nameinputs
╔
Ш
+__inference_dense_214_layer_call_fn_4116131

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCall▐
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
GPU2*0J 8В *O
fJRH
F__inference_dense_214_layer_call_and_return_conditional_losses_4114875o
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
Ц	
╙
8__inference_batch_normalization_71_layer_call_fn_4115960

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallЭ
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
GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_71_layer_call_and_return_conditional_losses_4114676Й
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
█
┤
-__inference_joint_model_layer_call_fn_4115316
	input_143
	input_144!
unknown:		
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:	

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:	Ф

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identityИвStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCall	input_143	input_144unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *8
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_joint_model_layer_call_and_return_conditional_losses_4115211o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:         Й:         ЙЙ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
0
_output_shapes
:         Й
#
_user_specified_name	input_143:\X
1
_output_shapes
:         ЙЙ
#
_user_specified_name	input_144
б

°
F__inference_dense_213_layer_call_and_return_conditional_losses_4116122

inputs1
matmul_readvariableop_resource:	Ф-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ф*
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
_construction_contextkEagerRuntime*+
_input_shapes
:         Ф: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         Ф
 
_user_specified_nameinputs
╠
d
H__inference_flatten_142_layer_call_and_return_conditional_losses_4114836

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"      ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ТY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         Т"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         Й:X T
0
_output_shapes
:         Й
 
_user_specified_nameinputs
Э

ў
F__inference_dense_214_layer_call_and_return_conditional_losses_4116142

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
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
А
p
Q__inference_spatial_dropout2d_71_layer_call_and_return_conditional_losses_4114620

inputs
identityИ;
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
valueB:╤
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
 *  а?З
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
value	B :╓
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:м
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
 *═╠L>╖
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"                  А
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"                  М
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
М
А
G__inference_conv2d_426_layer_call_and_return_conditional_losses_4115914

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ЙY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Йj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Йw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Й: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         Й
 
_user_specified_nameinputs
└
u
K__inference_concatenate_71_layer_call_and_return_conditional_losses_4114845

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :v
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:         ФX
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:         Ф"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         :         Т:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         Т
 
_user_specified_nameinputs
и
I
-__inference_flatten_143_layer_call_fn_4116072

inputs
identity╢
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
GPU2*0J 8В *Q
fLRJ
H__inference_flatten_143_layer_call_and_return_conditional_losses_4114828`
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
Н
А
G__inference_conv2d_430_layer_call_and_return_conditional_losses_4115934

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ы
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides	
Йr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ЙY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Йj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Йw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Й: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         Й
 
_user_specified_nameinputs
╩N
у
H__inference_joint_model_layer_call_and_return_conditional_losses_4114915

inputs
inputs_1,
conv2d_429_4114721:		 
conv2d_429_4114723:,
conv2d_426_4114738: 
conv2d_426_4114740:,
batch_normalization_71_4114744:,
batch_normalization_71_4114746:,
batch_normalization_71_4114748:,
batch_normalization_71_4114750:,
conv2d_430_4114765: 
conv2d_430_4114767:,
conv2d_427_4114782:	 
conv2d_427_4114784:,
conv2d_431_4114799: 
conv2d_431_4114801:,
conv2d_428_4114816: 
conv2d_428_4114818:$
dense_213_4114859:	Ф
dense_213_4114861:#
dense_214_4114876:
dense_214_4114878:#
dense_215_4114893:
dense_215_4114895:-
prediction_output_0_4114909:)
prediction_output_0_4114911:
identityИв.batch_normalization_71/StatefulPartitionedCallв"conv2d_426/StatefulPartitionedCallв"conv2d_427/StatefulPartitionedCallв"conv2d_428/StatefulPartitionedCallв"conv2d_429/StatefulPartitionedCallв"conv2d_430/StatefulPartitionedCallв"conv2d_431/StatefulPartitionedCallв!dense_213/StatefulPartitionedCallв!dense_214/StatefulPartitionedCallв!dense_215/StatefulPartitionedCallв+prediction_output_0/StatefulPartitionedCallЙ
"conv2d_429/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv2d_429_4114721conv2d_429_4114723*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_429_layer_call_and_return_conditional_losses_4114720З
"conv2d_426/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_426_4114738conv2d_426_4114740*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_426_layer_call_and_return_conditional_losses_4114737В
$spatial_dropout2d_71/PartitionedCallPartitionedCall+conv2d_429/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_spatial_dropout2d_71_layer_call_and_return_conditional_losses_4114592а
.batch_normalization_71/StatefulPartitionedCallStatefulPartitionedCall+conv2d_426/StatefulPartitionedCall:output:0batch_normalization_71_4114744batch_normalization_71_4114746batch_normalization_71_4114748batch_normalization_71_4114750*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_71_layer_call_and_return_conditional_losses_4114645о
"conv2d_430/StatefulPartitionedCallStatefulPartitionedCall-spatial_dropout2d_71/PartitionedCall:output:0conv2d_430_4114765conv2d_430_4114767*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_430_layer_call_and_return_conditional_losses_4114764╕
"conv2d_427/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_71/StatefulPartitionedCall:output:0conv2d_427_4114782conv2d_427_4114784*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_427_layer_call_and_return_conditional_losses_4114781м
"conv2d_431/StatefulPartitionedCallStatefulPartitionedCall+conv2d_430/StatefulPartitionedCall:output:0conv2d_431_4114799conv2d_431_4114801*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_431_layer_call_and_return_conditional_losses_4114798м
"conv2d_428/StatefulPartitionedCallStatefulPartitionedCall+conv2d_427/StatefulPartitionedCall:output:0conv2d_428_4114816conv2d_428_4114818*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_428_layer_call_and_return_conditional_losses_4114815 
'global_max_pooling2d_71/PartitionedCallPartitionedCall+conv2d_431/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *]
fXRV
T__inference_global_max_pooling2d_71_layer_call_and_return_conditional_losses_4114697ь
flatten_143/PartitionedCallPartitionedCall0global_max_pooling2d_71/PartitionedCall:output:0*
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
GPU2*0J 8В *Q
fLRJ
H__inference_flatten_143_layer_call_and_return_conditional_losses_4114828ш
flatten_142/PartitionedCallPartitionedCall+conv2d_428/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Т* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_flatten_142_layer_call_and_return_conditional_losses_4114836О
concatenate_71/PartitionedCallPartitionedCall$flatten_143/PartitionedCall:output:0$flatten_142/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ф* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_concatenate_71_layer_call_and_return_conditional_losses_4114845Ы
!dense_213/StatefulPartitionedCallStatefulPartitionedCall'concatenate_71/PartitionedCall:output:0dense_213_4114859dense_213_4114861*
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
GPU2*0J 8В *O
fJRH
F__inference_dense_213_layer_call_and_return_conditional_losses_4114858Ю
!dense_214/StatefulPartitionedCallStatefulPartitionedCall*dense_213/StatefulPartitionedCall:output:0dense_214_4114876dense_214_4114878*
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
GPU2*0J 8В *O
fJRH
F__inference_dense_214_layer_call_and_return_conditional_losses_4114875Ю
!dense_215/StatefulPartitionedCallStatefulPartitionedCall*dense_214/StatefulPartitionedCall:output:0dense_215_4114893dense_215_4114895*
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
GPU2*0J 8В *O
fJRH
F__inference_dense_215_layer_call_and_return_conditional_losses_4114892╞
+prediction_output_0/StatefulPartitionedCallStatefulPartitionedCall*dense_215/StatefulPartitionedCall:output:0prediction_output_0_4114909prediction_output_0_4114911*
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
GPU2*0J 8В *Y
fTRR
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4114908Г
IdentityIdentity4prediction_output_0/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         я
NoOpNoOp/^batch_normalization_71/StatefulPartitionedCall#^conv2d_426/StatefulPartitionedCall#^conv2d_427/StatefulPartitionedCall#^conv2d_428/StatefulPartitionedCall#^conv2d_429/StatefulPartitionedCall#^conv2d_430/StatefulPartitionedCall#^conv2d_431/StatefulPartitionedCall"^dense_213/StatefulPartitionedCall"^dense_214/StatefulPartitionedCall"^dense_215/StatefulPartitionedCall,^prediction_output_0/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:         Й:         ЙЙ: : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_71/StatefulPartitionedCall.batch_normalization_71/StatefulPartitionedCall2H
"conv2d_426/StatefulPartitionedCall"conv2d_426/StatefulPartitionedCall2H
"conv2d_427/StatefulPartitionedCall"conv2d_427/StatefulPartitionedCall2H
"conv2d_428/StatefulPartitionedCall"conv2d_428/StatefulPartitionedCall2H
"conv2d_429/StatefulPartitionedCall"conv2d_429/StatefulPartitionedCall2H
"conv2d_430/StatefulPartitionedCall"conv2d_430/StatefulPartitionedCall2H
"conv2d_431/StatefulPartitionedCall"conv2d_431/StatefulPartitionedCall2F
!dense_213/StatefulPartitionedCall!dense_213/StatefulPartitionedCall2F
!dense_214/StatefulPartitionedCall!dense_214/StatefulPartitionedCall2F
!dense_215/StatefulPartitionedCall!dense_215/StatefulPartitionedCall2Z
+prediction_output_0/StatefulPartitionedCall+prediction_output_0/StatefulPartitionedCall:X T
0
_output_shapes
:         Й
 
_user_specified_nameinputs:YU
1
_output_shapes
:         ЙЙ
 
_user_specified_nameinputs
М
А
G__inference_conv2d_426_layer_call_and_return_conditional_losses_4114737

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ЙY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Йj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Йw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Й: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         Й
 
_user_specified_nameinputs
╕
d
H__inference_flatten_143_layer_call_and_return_conditional_losses_4116078

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
И
┬
S__inference_batch_normalization_71_layer_call_and_return_conditional_losses_4115996

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ╘
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
╢Т
м
"__inference__wrapped_model_4114583
	input_143
	input_144O
5joint_model_conv2d_429_conv2d_readvariableop_resource:		D
6joint_model_conv2d_429_biasadd_readvariableop_resource:O
5joint_model_conv2d_426_conv2d_readvariableop_resource:D
6joint_model_conv2d_426_biasadd_readvariableop_resource:H
:joint_model_batch_normalization_71_readvariableop_resource:J
<joint_model_batch_normalization_71_readvariableop_1_resource:Y
Kjoint_model_batch_normalization_71_fusedbatchnormv3_readvariableop_resource:[
Mjoint_model_batch_normalization_71_fusedbatchnormv3_readvariableop_1_resource:O
5joint_model_conv2d_430_conv2d_readvariableop_resource:D
6joint_model_conv2d_430_biasadd_readvariableop_resource:O
5joint_model_conv2d_427_conv2d_readvariableop_resource:	D
6joint_model_conv2d_427_biasadd_readvariableop_resource:O
5joint_model_conv2d_431_conv2d_readvariableop_resource:D
6joint_model_conv2d_431_biasadd_readvariableop_resource:O
5joint_model_conv2d_428_conv2d_readvariableop_resource:D
6joint_model_conv2d_428_biasadd_readvariableop_resource:G
4joint_model_dense_213_matmul_readvariableop_resource:	ФC
5joint_model_dense_213_biasadd_readvariableop_resource:F
4joint_model_dense_214_matmul_readvariableop_resource:C
5joint_model_dense_214_biasadd_readvariableop_resource:F
4joint_model_dense_215_matmul_readvariableop_resource:C
5joint_model_dense_215_biasadd_readvariableop_resource:P
>joint_model_prediction_output_0_matmul_readvariableop_resource:M
?joint_model_prediction_output_0_biasadd_readvariableop_resource:
identityИвBjoint_model/batch_normalization_71/FusedBatchNormV3/ReadVariableOpвDjoint_model/batch_normalization_71/FusedBatchNormV3/ReadVariableOp_1в1joint_model/batch_normalization_71/ReadVariableOpв3joint_model/batch_normalization_71/ReadVariableOp_1в-joint_model/conv2d_426/BiasAdd/ReadVariableOpв,joint_model/conv2d_426/Conv2D/ReadVariableOpв-joint_model/conv2d_427/BiasAdd/ReadVariableOpв,joint_model/conv2d_427/Conv2D/ReadVariableOpв-joint_model/conv2d_428/BiasAdd/ReadVariableOpв,joint_model/conv2d_428/Conv2D/ReadVariableOpв-joint_model/conv2d_429/BiasAdd/ReadVariableOpв,joint_model/conv2d_429/Conv2D/ReadVariableOpв-joint_model/conv2d_430/BiasAdd/ReadVariableOpв,joint_model/conv2d_430/Conv2D/ReadVariableOpв-joint_model/conv2d_431/BiasAdd/ReadVariableOpв,joint_model/conv2d_431/Conv2D/ReadVariableOpв,joint_model/dense_213/BiasAdd/ReadVariableOpв+joint_model/dense_213/MatMul/ReadVariableOpв,joint_model/dense_214/BiasAdd/ReadVariableOpв+joint_model/dense_214/MatMul/ReadVariableOpв,joint_model/dense_215/BiasAdd/ReadVariableOpв+joint_model/dense_215/MatMul/ReadVariableOpв6joint_model/prediction_output_0/BiasAdd/ReadVariableOpв5joint_model/prediction_output_0/MatMul/ReadVariableOpк
,joint_model/conv2d_429/Conv2D/ReadVariableOpReadVariableOp5joint_model_conv2d_429_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0╠
joint_model/conv2d_429/Conv2DConv2D	input_1444joint_model/conv2d_429/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides	
Йа
-joint_model/conv2d_429/BiasAdd/ReadVariableOpReadVariableOp6joint_model_conv2d_429_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0├
joint_model/conv2d_429/BiasAddBiasAdd&joint_model/conv2d_429/Conv2D:output:05joint_model/conv2d_429/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ЙЗ
joint_model/conv2d_429/ReluRelu'joint_model/conv2d_429/BiasAdd:output:0*
T0*0
_output_shapes
:         Йк
,joint_model/conv2d_426/Conv2D/ReadVariableOpReadVariableOp5joint_model_conv2d_426_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╦
joint_model/conv2d_426/Conv2DConv2D	input_1434joint_model/conv2d_426/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides
а
-joint_model/conv2d_426/BiasAdd/ReadVariableOpReadVariableOp6joint_model_conv2d_426_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0├
joint_model/conv2d_426/BiasAddBiasAdd&joint_model/conv2d_426/Conv2D:output:05joint_model/conv2d_426/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ЙЗ
joint_model/conv2d_426/ReluRelu'joint_model/conv2d_426/BiasAdd:output:0*
T0*0
_output_shapes
:         ЙЫ
)joint_model/spatial_dropout2d_71/IdentityIdentity)joint_model/conv2d_429/Relu:activations:0*
T0*0
_output_shapes
:         Йи
1joint_model/batch_normalization_71/ReadVariableOpReadVariableOp:joint_model_batch_normalization_71_readvariableop_resource*
_output_shapes
:*
dtype0м
3joint_model/batch_normalization_71/ReadVariableOp_1ReadVariableOp<joint_model_batch_normalization_71_readvariableop_1_resource*
_output_shapes
:*
dtype0╩
Bjoint_model/batch_normalization_71/FusedBatchNormV3/ReadVariableOpReadVariableOpKjoint_model_batch_normalization_71_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╬
Djoint_model/batch_normalization_71/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMjoint_model_batch_normalization_71_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Й
3joint_model/batch_normalization_71/FusedBatchNormV3FusedBatchNormV3)joint_model/conv2d_426/Relu:activations:09joint_model/batch_normalization_71/ReadVariableOp:value:0;joint_model/batch_normalization_71/ReadVariableOp_1:value:0Jjoint_model/batch_normalization_71/FusedBatchNormV3/ReadVariableOp:value:0Ljoint_model/batch_normalization_71/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         Й:::::*
epsilon%oГ:*
is_training( к
,joint_model/conv2d_430/Conv2D/ReadVariableOpReadVariableOp5joint_model_conv2d_430_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ї
joint_model/conv2d_430/Conv2DConv2D2joint_model/spatial_dropout2d_71/Identity:output:04joint_model/conv2d_430/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides	
Йа
-joint_model/conv2d_430/BiasAdd/ReadVariableOpReadVariableOp6joint_model_conv2d_430_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0├
joint_model/conv2d_430/BiasAddBiasAdd&joint_model/conv2d_430/Conv2D:output:05joint_model/conv2d_430/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ЙЗ
joint_model/conv2d_430/ReluRelu'joint_model/conv2d_430/BiasAdd:output:0*
T0*0
_output_shapes
:         Йк
,joint_model/conv2d_427/Conv2D/ReadVariableOpReadVariableOp5joint_model_conv2d_427_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0∙
joint_model/conv2d_427/Conv2DConv2D7joint_model/batch_normalization_71/FusedBatchNormV3:y:04joint_model/conv2d_427/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides
а
-joint_model/conv2d_427/BiasAdd/ReadVariableOpReadVariableOp6joint_model_conv2d_427_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0├
joint_model/conv2d_427/BiasAddBiasAdd&joint_model/conv2d_427/Conv2D:output:05joint_model/conv2d_427/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ЙЗ
joint_model/conv2d_427/ReluRelu'joint_model/conv2d_427/BiasAdd:output:0*
T0*0
_output_shapes
:         Йк
,joint_model/conv2d_431/Conv2D/ReadVariableOpReadVariableOp5joint_model_conv2d_431_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ь
joint_model/conv2d_431/Conv2DConv2D)joint_model/conv2d_430/Relu:activations:04joint_model/conv2d_431/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides	
Йа
-joint_model/conv2d_431/BiasAdd/ReadVariableOpReadVariableOp6joint_model_conv2d_431_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0├
joint_model/conv2d_431/BiasAddBiasAdd&joint_model/conv2d_431/Conv2D:output:05joint_model/conv2d_431/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ЙЗ
joint_model/conv2d_431/ReluRelu'joint_model/conv2d_431/BiasAdd:output:0*
T0*0
_output_shapes
:         Йк
,joint_model/conv2d_428/Conv2D/ReadVariableOpReadVariableOp5joint_model_conv2d_428_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ы
joint_model/conv2d_428/Conv2DConv2D)joint_model/conv2d_427/Relu:activations:04joint_model/conv2d_428/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides
а
-joint_model/conv2d_428/BiasAdd/ReadVariableOpReadVariableOp6joint_model_conv2d_428_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0├
joint_model/conv2d_428/BiasAddBiasAdd&joint_model/conv2d_428/Conv2D:output:05joint_model/conv2d_428/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ЙЗ
joint_model/conv2d_428/ReluRelu'joint_model/conv2d_428/BiasAdd:output:0*
T0*0
_output_shapes
:         ЙК
9joint_model/global_max_pooling2d_71/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ╧
'joint_model/global_max_pooling2d_71/MaxMax)joint_model/conv2d_431/Relu:activations:0Bjoint_model/global_max_pooling2d_71/Max/reduction_indices:output:0*
T0*'
_output_shapes
:         n
joint_model/flatten_143/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ╢
joint_model/flatten_143/ReshapeReshape0joint_model/global_max_pooling2d_71/Max:output:0&joint_model/flatten_143/Const:output:0*
T0*'
_output_shapes
:         n
joint_model/flatten_142/ConstConst*
_output_shapes
:*
dtype0*
valueB"      ░
joint_model/flatten_142/ReshapeReshape)joint_model/conv2d_428/Relu:activations:0&joint_model/flatten_142/Const:output:0*
T0*(
_output_shapes
:         Тh
&joint_model/concatenate_71/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ю
!joint_model/concatenate_71/concatConcatV2(joint_model/flatten_143/Reshape:output:0(joint_model/flatten_142/Reshape:output:0/joint_model/concatenate_71/concat/axis:output:0*
N*
T0*(
_output_shapes
:         Фб
+joint_model/dense_213/MatMul/ReadVariableOpReadVariableOp4joint_model_dense_213_matmul_readvariableop_resource*
_output_shapes
:	Ф*
dtype0╣
joint_model/dense_213/MatMulMatMul*joint_model/concatenate_71/concat:output:03joint_model/dense_213/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ю
,joint_model/dense_213/BiasAdd/ReadVariableOpReadVariableOp5joint_model_dense_213_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╕
joint_model/dense_213/BiasAddBiasAdd&joint_model/dense_213/MatMul:product:04joint_model/dense_213/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         |
joint_model/dense_213/ReluRelu&joint_model/dense_213/BiasAdd:output:0*
T0*'
_output_shapes
:         а
+joint_model/dense_214/MatMul/ReadVariableOpReadVariableOp4joint_model_dense_214_matmul_readvariableop_resource*
_output_shapes

:*
dtype0╖
joint_model/dense_214/MatMulMatMul(joint_model/dense_213/Relu:activations:03joint_model/dense_214/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ю
,joint_model/dense_214/BiasAdd/ReadVariableOpReadVariableOp5joint_model_dense_214_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╕
joint_model/dense_214/BiasAddBiasAdd&joint_model/dense_214/MatMul:product:04joint_model/dense_214/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         |
joint_model/dense_214/ReluRelu&joint_model/dense_214/BiasAdd:output:0*
T0*'
_output_shapes
:         а
+joint_model/dense_215/MatMul/ReadVariableOpReadVariableOp4joint_model_dense_215_matmul_readvariableop_resource*
_output_shapes

:*
dtype0╖
joint_model/dense_215/MatMulMatMul(joint_model/dense_214/Relu:activations:03joint_model/dense_215/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ю
,joint_model/dense_215/BiasAdd/ReadVariableOpReadVariableOp5joint_model_dense_215_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╕
joint_model/dense_215/BiasAddBiasAdd&joint_model/dense_215/MatMul:product:04joint_model/dense_215/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         |
joint_model/dense_215/ReluRelu&joint_model/dense_215/BiasAdd:output:0*
T0*'
_output_shapes
:         ┤
5joint_model/prediction_output_0/MatMul/ReadVariableOpReadVariableOp>joint_model_prediction_output_0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0╦
&joint_model/prediction_output_0/MatMulMatMul(joint_model/dense_215/Relu:activations:0=joint_model/prediction_output_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ▓
6joint_model/prediction_output_0/BiasAdd/ReadVariableOpReadVariableOp?joint_model_prediction_output_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╓
'joint_model/prediction_output_0/BiasAddBiasAdd0joint_model/prediction_output_0/MatMul:product:0>joint_model/prediction_output_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
IdentityIdentity0joint_model/prediction_output_0/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ■	
NoOpNoOpC^joint_model/batch_normalization_71/FusedBatchNormV3/ReadVariableOpE^joint_model/batch_normalization_71/FusedBatchNormV3/ReadVariableOp_12^joint_model/batch_normalization_71/ReadVariableOp4^joint_model/batch_normalization_71/ReadVariableOp_1.^joint_model/conv2d_426/BiasAdd/ReadVariableOp-^joint_model/conv2d_426/Conv2D/ReadVariableOp.^joint_model/conv2d_427/BiasAdd/ReadVariableOp-^joint_model/conv2d_427/Conv2D/ReadVariableOp.^joint_model/conv2d_428/BiasAdd/ReadVariableOp-^joint_model/conv2d_428/Conv2D/ReadVariableOp.^joint_model/conv2d_429/BiasAdd/ReadVariableOp-^joint_model/conv2d_429/Conv2D/ReadVariableOp.^joint_model/conv2d_430/BiasAdd/ReadVariableOp-^joint_model/conv2d_430/Conv2D/ReadVariableOp.^joint_model/conv2d_431/BiasAdd/ReadVariableOp-^joint_model/conv2d_431/Conv2D/ReadVariableOp-^joint_model/dense_213/BiasAdd/ReadVariableOp,^joint_model/dense_213/MatMul/ReadVariableOp-^joint_model/dense_214/BiasAdd/ReadVariableOp,^joint_model/dense_214/MatMul/ReadVariableOp-^joint_model/dense_215/BiasAdd/ReadVariableOp,^joint_model/dense_215/MatMul/ReadVariableOp7^joint_model/prediction_output_0/BiasAdd/ReadVariableOp6^joint_model/prediction_output_0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:         Й:         ЙЙ: : : : : : : : : : : : : : : : : : : : : : : : 2И
Bjoint_model/batch_normalization_71/FusedBatchNormV3/ReadVariableOpBjoint_model/batch_normalization_71/FusedBatchNormV3/ReadVariableOp2М
Djoint_model/batch_normalization_71/FusedBatchNormV3/ReadVariableOp_1Djoint_model/batch_normalization_71/FusedBatchNormV3/ReadVariableOp_12f
1joint_model/batch_normalization_71/ReadVariableOp1joint_model/batch_normalization_71/ReadVariableOp2j
3joint_model/batch_normalization_71/ReadVariableOp_13joint_model/batch_normalization_71/ReadVariableOp_12^
-joint_model/conv2d_426/BiasAdd/ReadVariableOp-joint_model/conv2d_426/BiasAdd/ReadVariableOp2\
,joint_model/conv2d_426/Conv2D/ReadVariableOp,joint_model/conv2d_426/Conv2D/ReadVariableOp2^
-joint_model/conv2d_427/BiasAdd/ReadVariableOp-joint_model/conv2d_427/BiasAdd/ReadVariableOp2\
,joint_model/conv2d_427/Conv2D/ReadVariableOp,joint_model/conv2d_427/Conv2D/ReadVariableOp2^
-joint_model/conv2d_428/BiasAdd/ReadVariableOp-joint_model/conv2d_428/BiasAdd/ReadVariableOp2\
,joint_model/conv2d_428/Conv2D/ReadVariableOp,joint_model/conv2d_428/Conv2D/ReadVariableOp2^
-joint_model/conv2d_429/BiasAdd/ReadVariableOp-joint_model/conv2d_429/BiasAdd/ReadVariableOp2\
,joint_model/conv2d_429/Conv2D/ReadVariableOp,joint_model/conv2d_429/Conv2D/ReadVariableOp2^
-joint_model/conv2d_430/BiasAdd/ReadVariableOp-joint_model/conv2d_430/BiasAdd/ReadVariableOp2\
,joint_model/conv2d_430/Conv2D/ReadVariableOp,joint_model/conv2d_430/Conv2D/ReadVariableOp2^
-joint_model/conv2d_431/BiasAdd/ReadVariableOp-joint_model/conv2d_431/BiasAdd/ReadVariableOp2\
,joint_model/conv2d_431/Conv2D/ReadVariableOp,joint_model/conv2d_431/Conv2D/ReadVariableOp2\
,joint_model/dense_213/BiasAdd/ReadVariableOp,joint_model/dense_213/BiasAdd/ReadVariableOp2Z
+joint_model/dense_213/MatMul/ReadVariableOp+joint_model/dense_213/MatMul/ReadVariableOp2\
,joint_model/dense_214/BiasAdd/ReadVariableOp,joint_model/dense_214/BiasAdd/ReadVariableOp2Z
+joint_model/dense_214/MatMul/ReadVariableOp+joint_model/dense_214/MatMul/ReadVariableOp2\
,joint_model/dense_215/BiasAdd/ReadVariableOp,joint_model/dense_215/BiasAdd/ReadVariableOp2Z
+joint_model/dense_215/MatMul/ReadVariableOp+joint_model/dense_215/MatMul/ReadVariableOp2p
6joint_model/prediction_output_0/BiasAdd/ReadVariableOp6joint_model/prediction_output_0/BiasAdd/ReadVariableOp2n
5joint_model/prediction_output_0/MatMul/ReadVariableOp5joint_model/prediction_output_0/MatMul/ReadVariableOp:[ W
0
_output_shapes
:         Й
#
_user_specified_name	input_143:\X
1
_output_shapes
:         ЙЙ
#
_user_specified_name	input_144
╠
d
H__inference_flatten_142_layer_call_and_return_conditional_losses_4116089

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"      ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ТY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         Т"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         Й:X T
0
_output_shapes
:         Й
 
_user_specified_nameinputs
ў
б
,__inference_conv2d_430_layer_call_fn_4115923

inputs!
unknown:
	unknown_0:
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_430_layer_call_and_return_conditional_losses_4114764x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         Й`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Й: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         Й
 
_user_specified_nameinputs
Э

ў
F__inference_dense_215_layer_call_and_return_conditional_losses_4116162

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
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
╙	
Б
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4116181

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
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
ў
б
,__inference_conv2d_426_layer_call_fn_4115903

inputs!
unknown:
	unknown_0:
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_426_layer_call_and_return_conditional_losses_4114737x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         Й`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Й: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         Й
 
_user_specified_nameinputs
Н
А
G__inference_conv2d_431_layer_call_and_return_conditional_losses_4116016

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ы
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides	
Йr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ЙY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Йj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Йw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         Й: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         Й
 
_user_specified_nameinputs
╬Ш
т
H__inference_joint_model_layer_call_and_return_conditional_losses_4115836
inputs_0
inputs_1C
)conv2d_429_conv2d_readvariableop_resource:		8
*conv2d_429_biasadd_readvariableop_resource:C
)conv2d_426_conv2d_readvariableop_resource:8
*conv2d_426_biasadd_readvariableop_resource:<
.batch_normalization_71_readvariableop_resource:>
0batch_normalization_71_readvariableop_1_resource:M
?batch_normalization_71_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_71_fusedbatchnormv3_readvariableop_1_resource:C
)conv2d_430_conv2d_readvariableop_resource:8
*conv2d_430_biasadd_readvariableop_resource:C
)conv2d_427_conv2d_readvariableop_resource:	8
*conv2d_427_biasadd_readvariableop_resource:C
)conv2d_431_conv2d_readvariableop_resource:8
*conv2d_431_biasadd_readvariableop_resource:C
)conv2d_428_conv2d_readvariableop_resource:8
*conv2d_428_biasadd_readvariableop_resource:;
(dense_213_matmul_readvariableop_resource:	Ф7
)dense_213_biasadd_readvariableop_resource::
(dense_214_matmul_readvariableop_resource:7
)dense_214_biasadd_readvariableop_resource::
(dense_215_matmul_readvariableop_resource:7
)dense_215_biasadd_readvariableop_resource:D
2prediction_output_0_matmul_readvariableop_resource:A
3prediction_output_0_biasadd_readvariableop_resource:
identityИв%batch_normalization_71/AssignNewValueв'batch_normalization_71/AssignNewValue_1в6batch_normalization_71/FusedBatchNormV3/ReadVariableOpв8batch_normalization_71/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_71/ReadVariableOpв'batch_normalization_71/ReadVariableOp_1в!conv2d_426/BiasAdd/ReadVariableOpв conv2d_426/Conv2D/ReadVariableOpв!conv2d_427/BiasAdd/ReadVariableOpв conv2d_427/Conv2D/ReadVariableOpв!conv2d_428/BiasAdd/ReadVariableOpв conv2d_428/Conv2D/ReadVariableOpв!conv2d_429/BiasAdd/ReadVariableOpв conv2d_429/Conv2D/ReadVariableOpв!conv2d_430/BiasAdd/ReadVariableOpв conv2d_430/Conv2D/ReadVariableOpв!conv2d_431/BiasAdd/ReadVariableOpв conv2d_431/Conv2D/ReadVariableOpв dense_213/BiasAdd/ReadVariableOpвdense_213/MatMul/ReadVariableOpв dense_214/BiasAdd/ReadVariableOpвdense_214/MatMul/ReadVariableOpв dense_215/BiasAdd/ReadVariableOpвdense_215/MatMul/ReadVariableOpв*prediction_output_0/BiasAdd/ReadVariableOpв)prediction_output_0/MatMul/ReadVariableOpТ
 conv2d_429/Conv2D/ReadVariableOpReadVariableOp)conv2d_429_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0│
conv2d_429/Conv2DConv2Dinputs_1(conv2d_429/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides	
ЙИ
!conv2d_429/BiasAdd/ReadVariableOpReadVariableOp*conv2d_429_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv2d_429/BiasAddBiasAddconv2d_429/Conv2D:output:0)conv2d_429/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Йo
conv2d_429/ReluReluconv2d_429/BiasAdd:output:0*
T0*0
_output_shapes
:         ЙТ
 conv2d_426/Conv2D/ReadVariableOpReadVariableOp)conv2d_426_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0▓
conv2d_426/Conv2DConv2Dinputs_0(conv2d_426/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides
И
!conv2d_426/BiasAdd/ReadVariableOpReadVariableOp*conv2d_426_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv2d_426/BiasAddBiasAddconv2d_426/Conv2D:output:0)conv2d_426/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Йo
conv2d_426/ReluReluconv2d_426/BiasAdd:output:0*
T0*0
_output_shapes
:         Йg
spatial_dropout2d_71/ShapeShapeconv2d_429/Relu:activations:0*
T0*
_output_shapes
:r
(spatial_dropout2d_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*spatial_dropout2d_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*spatial_dropout2d_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"spatial_dropout2d_71/strided_sliceStridedSlice#spatial_dropout2d_71/Shape:output:01spatial_dropout2d_71/strided_slice/stack:output:03spatial_dropout2d_71/strided_slice/stack_1:output:03spatial_dropout2d_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
*spatial_dropout2d_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,spatial_dropout2d_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,spatial_dropout2d_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┬
$spatial_dropout2d_71/strided_slice_1StridedSlice#spatial_dropout2d_71/Shape:output:03spatial_dropout2d_71/strided_slice_1/stack:output:05spatial_dropout2d_71/strided_slice_1/stack_1:output:05spatial_dropout2d_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
"spatial_dropout2d_71/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?о
 spatial_dropout2d_71/dropout/MulMulconv2d_429/Relu:activations:0+spatial_dropout2d_71/dropout/Const:output:0*
T0*0
_output_shapes
:         Йu
3spatial_dropout2d_71/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :u
3spatial_dropout2d_71/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :┐
1spatial_dropout2d_71/dropout/random_uniform/shapePack+spatial_dropout2d_71/strided_slice:output:0<spatial_dropout2d_71/dropout/random_uniform/shape/1:output:0<spatial_dropout2d_71/dropout/random_uniform/shape/2:output:0-spatial_dropout2d_71/strided_slice_1:output:0*
N*
T0*
_output_shapes
:═
9spatial_dropout2d_71/dropout/random_uniform/RandomUniformRandomUniform:spatial_dropout2d_71/dropout/random_uniform/shape:output:0*
T0*/
_output_shapes
:         *
dtype0p
+spatial_dropout2d_71/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>э
)spatial_dropout2d_71/dropout/GreaterEqualGreaterEqualBspatial_dropout2d_71/dropout/random_uniform/RandomUniform:output:04spatial_dropout2d_71/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         б
!spatial_dropout2d_71/dropout/CastCast-spatial_dropout2d_71/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         ▒
"spatial_dropout2d_71/dropout/Mul_1Mul$spatial_dropout2d_71/dropout/Mul:z:0%spatial_dropout2d_71/dropout/Cast:y:0*
T0*0
_output_shapes
:         ЙР
%batch_normalization_71/ReadVariableOpReadVariableOp.batch_normalization_71_readvariableop_resource*
_output_shapes
:*
dtype0Ф
'batch_normalization_71/ReadVariableOp_1ReadVariableOp0batch_normalization_71_readvariableop_1_resource*
_output_shapes
:*
dtype0▓
6batch_normalization_71/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_71_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╢
8batch_normalization_71/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_71_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╧
'batch_normalization_71/FusedBatchNormV3FusedBatchNormV3conv2d_426/Relu:activations:0-batch_normalization_71/ReadVariableOp:value:0/batch_normalization_71/ReadVariableOp_1:value:0>batch_normalization_71/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_71/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         Й:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<в
%batch_normalization_71/AssignNewValueAssignVariableOp?batch_normalization_71_fusedbatchnormv3_readvariableop_resource4batch_normalization_71/FusedBatchNormV3:batch_mean:07^batch_normalization_71/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(м
'batch_normalization_71/AssignNewValue_1AssignVariableOpAbatch_normalization_71_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_71/FusedBatchNormV3:batch_variance:09^batch_normalization_71/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Т
 conv2d_430/Conv2D/ReadVariableOpReadVariableOp)conv2d_430_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╤
conv2d_430/Conv2DConv2D&spatial_dropout2d_71/dropout/Mul_1:z:0(conv2d_430/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides	
ЙИ
!conv2d_430/BiasAdd/ReadVariableOpReadVariableOp*conv2d_430_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv2d_430/BiasAddBiasAddconv2d_430/Conv2D:output:0)conv2d_430/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Йo
conv2d_430/ReluReluconv2d_430/BiasAdd:output:0*
T0*0
_output_shapes
:         ЙТ
 conv2d_427/Conv2D/ReadVariableOpReadVariableOp)conv2d_427_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0╒
conv2d_427/Conv2DConv2D+batch_normalization_71/FusedBatchNormV3:y:0(conv2d_427/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides
И
!conv2d_427/BiasAdd/ReadVariableOpReadVariableOp*conv2d_427_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv2d_427/BiasAddBiasAddconv2d_427/Conv2D:output:0)conv2d_427/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Йo
conv2d_427/ReluReluconv2d_427/BiasAdd:output:0*
T0*0
_output_shapes
:         ЙТ
 conv2d_431/Conv2D/ReadVariableOpReadVariableOp)conv2d_431_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╚
conv2d_431/Conv2DConv2Dconv2d_430/Relu:activations:0(conv2d_431/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides	
ЙИ
!conv2d_431/BiasAdd/ReadVariableOpReadVariableOp*conv2d_431_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv2d_431/BiasAddBiasAddconv2d_431/Conv2D:output:0)conv2d_431/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Йo
conv2d_431/ReluReluconv2d_431/BiasAdd:output:0*
T0*0
_output_shapes
:         ЙТ
 conv2d_428/Conv2D/ReadVariableOpReadVariableOp)conv2d_428_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╟
conv2d_428/Conv2DConv2Dconv2d_427/Relu:activations:0(conv2d_428/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides
И
!conv2d_428/BiasAdd/ReadVariableOpReadVariableOp*conv2d_428_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv2d_428/BiasAddBiasAddconv2d_428/Conv2D:output:0)conv2d_428/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Йo
conv2d_428/ReluReluconv2d_428/BiasAdd:output:0*
T0*0
_output_shapes
:         Й~
-global_max_pooling2d_71/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      л
global_max_pooling2d_71/MaxMaxconv2d_431/Relu:activations:06global_max_pooling2d_71/Max/reduction_indices:output:0*
T0*'
_output_shapes
:         b
flatten_143/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Т
flatten_143/ReshapeReshape$global_max_pooling2d_71/Max:output:0flatten_143/Const:output:0*
T0*'
_output_shapes
:         b
flatten_142/ConstConst*
_output_shapes
:*
dtype0*
valueB"      М
flatten_142/ReshapeReshapeconv2d_428/Relu:activations:0flatten_142/Const:output:0*
T0*(
_output_shapes
:         Т\
concatenate_71/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :╛
concatenate_71/concatConcatV2flatten_143/Reshape:output:0flatten_142/Reshape:output:0#concatenate_71/concat/axis:output:0*
N*
T0*(
_output_shapes
:         ФЙ
dense_213/MatMul/ReadVariableOpReadVariableOp(dense_213_matmul_readvariableop_resource*
_output_shapes
:	Ф*
dtype0Х
dense_213/MatMulMatMulconcatenate_71/concat:output:0'dense_213/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ж
 dense_213/BiasAdd/ReadVariableOpReadVariableOp)dense_213_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
dense_213/BiasAddBiasAdddense_213/MatMul:product:0(dense_213/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d
dense_213/ReluReludense_213/BiasAdd:output:0*
T0*'
_output_shapes
:         И
dense_214/MatMul/ReadVariableOpReadVariableOp(dense_214_matmul_readvariableop_resource*
_output_shapes

:*
dtype0У
dense_214/MatMulMatMuldense_213/Relu:activations:0'dense_214/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ж
 dense_214/BiasAdd/ReadVariableOpReadVariableOp)dense_214_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
dense_214/BiasAddBiasAdddense_214/MatMul:product:0(dense_214/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d
dense_214/ReluReludense_214/BiasAdd:output:0*
T0*'
_output_shapes
:         И
dense_215/MatMul/ReadVariableOpReadVariableOp(dense_215_matmul_readvariableop_resource*
_output_shapes

:*
dtype0У
dense_215/MatMulMatMuldense_214/Relu:activations:0'dense_215/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ж
 dense_215/BiasAdd/ReadVariableOpReadVariableOp)dense_215_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
dense_215/BiasAddBiasAdddense_215/MatMul:product:0(dense_215/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d
dense_215/ReluReludense_215/BiasAdd:output:0*
T0*'
_output_shapes
:         Ь
)prediction_output_0/MatMul/ReadVariableOpReadVariableOp2prediction_output_0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0з
prediction_output_0/MatMulMatMuldense_215/Relu:activations:01prediction_output_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ъ
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
:         ░
NoOpNoOp&^batch_normalization_71/AssignNewValue(^batch_normalization_71/AssignNewValue_17^batch_normalization_71/FusedBatchNormV3/ReadVariableOp9^batch_normalization_71/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_71/ReadVariableOp(^batch_normalization_71/ReadVariableOp_1"^conv2d_426/BiasAdd/ReadVariableOp!^conv2d_426/Conv2D/ReadVariableOp"^conv2d_427/BiasAdd/ReadVariableOp!^conv2d_427/Conv2D/ReadVariableOp"^conv2d_428/BiasAdd/ReadVariableOp!^conv2d_428/Conv2D/ReadVariableOp"^conv2d_429/BiasAdd/ReadVariableOp!^conv2d_429/Conv2D/ReadVariableOp"^conv2d_430/BiasAdd/ReadVariableOp!^conv2d_430/Conv2D/ReadVariableOp"^conv2d_431/BiasAdd/ReadVariableOp!^conv2d_431/Conv2D/ReadVariableOp!^dense_213/BiasAdd/ReadVariableOp ^dense_213/MatMul/ReadVariableOp!^dense_214/BiasAdd/ReadVariableOp ^dense_214/MatMul/ReadVariableOp!^dense_215/BiasAdd/ReadVariableOp ^dense_215/MatMul/ReadVariableOp+^prediction_output_0/BiasAdd/ReadVariableOp*^prediction_output_0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:         Й:         ЙЙ: : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_71/AssignNewValue%batch_normalization_71/AssignNewValue2R
'batch_normalization_71/AssignNewValue_1'batch_normalization_71/AssignNewValue_12p
6batch_normalization_71/FusedBatchNormV3/ReadVariableOp6batch_normalization_71/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_71/FusedBatchNormV3/ReadVariableOp_18batch_normalization_71/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_71/ReadVariableOp%batch_normalization_71/ReadVariableOp2R
'batch_normalization_71/ReadVariableOp_1'batch_normalization_71/ReadVariableOp_12F
!conv2d_426/BiasAdd/ReadVariableOp!conv2d_426/BiasAdd/ReadVariableOp2D
 conv2d_426/Conv2D/ReadVariableOp conv2d_426/Conv2D/ReadVariableOp2F
!conv2d_427/BiasAdd/ReadVariableOp!conv2d_427/BiasAdd/ReadVariableOp2D
 conv2d_427/Conv2D/ReadVariableOp conv2d_427/Conv2D/ReadVariableOp2F
!conv2d_428/BiasAdd/ReadVariableOp!conv2d_428/BiasAdd/ReadVariableOp2D
 conv2d_428/Conv2D/ReadVariableOp conv2d_428/Conv2D/ReadVariableOp2F
!conv2d_429/BiasAdd/ReadVariableOp!conv2d_429/BiasAdd/ReadVariableOp2D
 conv2d_429/Conv2D/ReadVariableOp conv2d_429/Conv2D/ReadVariableOp2F
!conv2d_430/BiasAdd/ReadVariableOp!conv2d_430/BiasAdd/ReadVariableOp2D
 conv2d_430/Conv2D/ReadVariableOp conv2d_430/Conv2D/ReadVariableOp2F
!conv2d_431/BiasAdd/ReadVariableOp!conv2d_431/BiasAdd/ReadVariableOp2D
 conv2d_431/Conv2D/ReadVariableOp conv2d_431/Conv2D/ReadVariableOp2D
 dense_213/BiasAdd/ReadVariableOp dense_213/BiasAdd/ReadVariableOp2B
dense_213/MatMul/ReadVariableOpdense_213/MatMul/ReadVariableOp2D
 dense_214/BiasAdd/ReadVariableOp dense_214/BiasAdd/ReadVariableOp2B
dense_214/MatMul/ReadVariableOpdense_214/MatMul/ReadVariableOp2D
 dense_215/BiasAdd/ReadVariableOp dense_215/BiasAdd/ReadVariableOp2B
dense_215/MatMul/ReadVariableOpdense_215/MatMul/ReadVariableOp2X
*prediction_output_0/BiasAdd/ReadVariableOp*prediction_output_0/BiasAdd/ReadVariableOp2V
)prediction_output_0/MatMul/ReadVariableOp)prediction_output_0/MatMul/ReadVariableOp:Z V
0
_output_shapes
:         Й
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:         ЙЙ
"
_user_specified_name
inputs/1
╔
Ш
+__inference_dense_215_layer_call_fn_4116151

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCall▐
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
GPU2*0J 8В *O
fJRH
F__inference_dense_215_layer_call_and_return_conditional_losses_4114892o
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
Ш
U
9__inference_global_max_pooling2d_71_layer_call_fn_4116041

inputs
identity╦
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
GPU2*0J 8В *]
fXRV
T__inference_global_max_pooling2d_71_layer_call_and_return_conditional_losses_4114697i
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
╫
▓
-__inference_joint_model_layer_call_fn_4115570
inputs_0
inputs_1!
unknown:		
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:	

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:	Ф

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identityИвStatefulPartitionedCallШ
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
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_joint_model_layer_call_and_return_conditional_losses_4114915o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:         Й:         ЙЙ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:         Й
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:         ЙЙ
"
_user_specified_name
inputs/1
п
м
%__inference_signature_wrapper_4115516
	input_143
	input_144!
unknown:		
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:	

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:	Ф

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:
identityИвStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCall	input_143	input_144unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__wrapped_model_4114583o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:         Й:         ЙЙ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
0
_output_shapes
:         Й
#
_user_specified_name	input_143:\X
1
_output_shapes
:         ЙЙ
#
_user_specified_name	input_144"╡	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*О
serving_default·
H
	input_143;
serving_default_input_143:0         Й
I
	input_144<
serving_default_input_144:0         ЙЙG
prediction_output_00
StatefulPartitionedCall:0         tensorflow/serving/predict:╥Й
Ї
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
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
▌
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias
 $_jit_compiled_convolution_op"
_tf_keras_layer
"
_tf_keras_input_layer
╝
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
+_random_generator"
_tf_keras_layer
▌
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias
 4_jit_compiled_convolution_op"
_tf_keras_layer
▌
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias
 =_jit_compiled_convolution_op"
_tf_keras_layer
ъ
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses
Daxis
	Egamma
Fbeta
Gmoving_mean
Hmoving_variance"
_tf_keras_layer
▌
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses

Okernel
Pbias
 Q_jit_compiled_convolution_op"
_tf_keras_layer
▌
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses

Xkernel
Ybias
 Z_jit_compiled_convolution_op"
_tf_keras_layer
е
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses

gkernel
hbias
 i_jit_compiled_convolution_op"
_tf_keras_layer
е
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses"
_tf_keras_layer
е
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
е
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
┐
|	variables
}trainable_variables
~regularization_losses
	keras_api
А__call__
+Б&call_and_return_all_conditional_losses
Вkernel
	Гbias"
_tf_keras_layer
├
Д	variables
Еtrainable_variables
Жregularization_losses
З	keras_api
И__call__
+Й&call_and_return_all_conditional_losses
Кkernel
	Лbias"
_tf_keras_layer
├
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
Р__call__
+С&call_and_return_all_conditional_losses
Тkernel
	Уbias"
_tf_keras_layer
├
Ф	variables
Хtrainable_variables
Цregularization_losses
Ч	keras_api
Ш__call__
+Щ&call_and_return_all_conditional_losses
Ъkernel
	Ыbias"
_tf_keras_layer
▐
"0
#1
22
33
;4
<5
E6
F7
G8
H9
O10
P11
X12
Y13
g14
h15
В16
Г17
К18
Л19
Т20
У21
Ъ22
Ы23"
trackable_list_wrapper
╬
"0
#1
22
33
;4
<5
E6
F7
O8
P9
X10
Y11
g12
h13
В14
Г15
К16
Л17
Т18
У19
Ъ20
Ы21"
trackable_list_wrapper
 "
trackable_list_wrapper
╧
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ё
бtrace_0
вtrace_1
гtrace_2
дtrace_32■
-__inference_joint_model_layer_call_fn_4114966
-__inference_joint_model_layer_call_fn_4115570
-__inference_joint_model_layer_call_fn_4115624
-__inference_joint_model_layer_call_fn_4115316┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zбtrace_0zвtrace_1zгtrace_2zдtrace_3
▌
еtrace_0
жtrace_1
зtrace_2
иtrace_32ъ
H__inference_joint_model_layer_call_and_return_conditional_losses_4115721
H__inference_joint_model_layer_call_and_return_conditional_losses_4115836
H__inference_joint_model_layer_call_and_return_conditional_losses_4115385
H__inference_joint_model_layer_call_and_return_conditional_losses_4115454┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zеtrace_0zжtrace_1zзtrace_2zиtrace_3
┌B╫
"__inference__wrapped_model_4114583	input_143	input_144"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
а
йbeta_1
кbeta_2

лdecay
мlearning_rate
	нiter"mи#mй2mк3mл;mм<mнEmоFmпOm░Pm▒Xm▓Ym│gm┤hm╡	Вm╢	Гm╖	Кm╕	Лm╣	Тm║	Уm╗	Ъm╝	Ыm╜"v╛#v┐2v└3v┴;v┬<v├Ev─Fv┼Ov╞Pv╟Xv╚Yv╔gv╩hv╦	Вv╠	Гv═	Кv╬	Лv╧	Тv╨	Уv╤	Ъv╥	Ыv╙"
	optimizer
-
оserving_default"
signature_map
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
пnon_trainable_variables
░layers
▒metrics
 ▓layer_regularization_losses
│layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
Є
┤trace_02╙
,__inference_conv2d_429_layer_call_fn_4115845в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┤trace_0
Н
╡trace_02ю
G__inference_conv2d_429_layer_call_and_return_conditional_losses_4115856в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╡trace_0
+:)		2conv2d_429/kernel
:2conv2d_429/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╢non_trainable_variables
╖layers
╕metrics
 ╣layer_regularization_losses
║layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
с
╗trace_0
╝trace_12ж
6__inference_spatial_dropout2d_71_layer_call_fn_4115861
6__inference_spatial_dropout2d_71_layer_call_fn_4115866│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╗trace_0z╝trace_1
Ч
╜trace_0
╛trace_12▄
Q__inference_spatial_dropout2d_71_layer_call_and_return_conditional_losses_4115871
Q__inference_spatial_dropout2d_71_layer_call_and_return_conditional_losses_4115894│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╜trace_0z╛trace_1
"
_generic_user_object
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
┐non_trainable_variables
└layers
┴metrics
 ┬layer_regularization_losses
├layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
Є
─trace_02╙
,__inference_conv2d_426_layer_call_fn_4115903в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z─trace_0
Н
┼trace_02ю
G__inference_conv2d_426_layer_call_and_return_conditional_losses_4115914в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┼trace_0
+:)2conv2d_426/kernel
:2conv2d_426/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╞non_trainable_variables
╟layers
╚metrics
 ╔layer_regularization_losses
╩layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
Є
╦trace_02╙
,__inference_conv2d_430_layer_call_fn_4115923в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╦trace_0
Н
╠trace_02ю
G__inference_conv2d_430_layer_call_and_return_conditional_losses_4115934в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╠trace_0
+:)2conv2d_430/kernel
:2conv2d_430/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
<
E0
F1
G2
H3"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
═non_trainable_variables
╬layers
╧metrics
 ╨layer_regularization_losses
╤layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
х
╥trace_0
╙trace_12к
8__inference_batch_normalization_71_layer_call_fn_4115947
8__inference_batch_normalization_71_layer_call_fn_4115960│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╥trace_0z╙trace_1
Ы
╘trace_0
╒trace_12р
S__inference_batch_normalization_71_layer_call_and_return_conditional_losses_4115978
S__inference_batch_normalization_71_layer_call_and_return_conditional_losses_4115996│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╘trace_0z╒trace_1
 "
trackable_list_wrapper
*:(2batch_normalization_71/gamma
):'2batch_normalization_71/beta
2:0 (2"batch_normalization_71/moving_mean
6:4 (2&batch_normalization_71/moving_variance
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╓non_trainable_variables
╫layers
╪metrics
 ┘layer_regularization_losses
┌layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
Є
█trace_02╙
,__inference_conv2d_431_layer_call_fn_4116005в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z█trace_0
Н
▄trace_02ю
G__inference_conv2d_431_layer_call_and_return_conditional_losses_4116016в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▄trace_0
+:)2conv2d_431/kernel
:2conv2d_431/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
▌non_trainable_variables
▐layers
▀metrics
 рlayer_regularization_losses
сlayer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
Є
тtrace_02╙
,__inference_conv2d_427_layer_call_fn_4116025в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zтtrace_0
Н
уtrace_02ю
G__inference_conv2d_427_layer_call_and_return_conditional_losses_4116036в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zуtrace_0
+:)	2conv2d_427/kernel
:2conv2d_427/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
фnon_trainable_variables
хlayers
цmetrics
 чlayer_regularization_losses
шlayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
 
щtrace_02р
9__inference_global_max_pooling2d_71_layer_call_fn_4116041в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zщtrace_0
Ъ
ъtrace_02√
T__inference_global_max_pooling2d_71_layer_call_and_return_conditional_losses_4116047в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zъtrace_0
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
яlayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
Є
Ёtrace_02╙
,__inference_conv2d_428_layer_call_fn_4116056в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЁtrace_0
Н
ёtrace_02ю
G__inference_conv2d_428_layer_call_and_return_conditional_losses_4116067в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zёtrace_0
+:)2conv2d_428/kernel
:2conv2d_428/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Єnon_trainable_variables
єlayers
Їmetrics
 їlayer_regularization_losses
Ўlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
є
ўtrace_02╘
-__inference_flatten_143_layer_call_fn_4116072в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zўtrace_0
О
°trace_02я
H__inference_flatten_143_layer_call_and_return_conditional_losses_4116078в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z°trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
∙non_trainable_variables
·layers
√metrics
 №layer_regularization_losses
¤layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
є
■trace_02╘
-__inference_flatten_142_layer_call_fn_4116083в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z■trace_0
О
 trace_02я
H__inference_flatten_142_layer_call_and_return_conditional_losses_4116089в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
Ў
Еtrace_02╫
0__inference_concatenate_71_layer_call_fn_4116095в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЕtrace_0
С
Жtrace_02Є
K__inference_concatenate_71_layer_call_and_return_conditional_losses_4116102в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЖtrace_0
0
В0
Г1"
trackable_list_wrapper
0
В0
Г1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
|	variables
}trainable_variables
~regularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
ё
Мtrace_02╥
+__inference_dense_213_layer_call_fn_4116111в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zМtrace_0
М
Нtrace_02э
F__inference_dense_213_layer_call_and_return_conditional_losses_4116122в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zНtrace_0
#:!	Ф2dense_213/kernel
:2dense_213/bias
0
К0
Л1"
trackable_list_wrapper
0
К0
Л1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
Д	variables
Еtrainable_variables
Жregularization_losses
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
ё
Уtrace_02╥
+__inference_dense_214_layer_call_fn_4116131в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zУtrace_0
М
Фtrace_02э
F__inference_dense_214_layer_call_and_return_conditional_losses_4116142в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zФtrace_0
": 2dense_214/kernel
:2dense_214/bias
0
Т0
У1"
trackable_list_wrapper
0
Т0
У1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
М	variables
Нtrainable_variables
Оregularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
ё
Ъtrace_02╥
+__inference_dense_215_layer_call_fn_4116151в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЪtrace_0
М
Ыtrace_02э
F__inference_dense_215_layer_call_and_return_conditional_losses_4116162в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЫtrace_0
": 2dense_215/kernel
:2dense_215/bias
0
Ъ0
Ы1"
trackable_list_wrapper
0
Ъ0
Ы1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
Ф	variables
Хtrainable_variables
Цregularization_losses
Ш__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
√
бtrace_02▄
5__inference_prediction_output_0_layer_call_fn_4116171в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zбtrace_0
Ц
вtrace_02ў
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4116181в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zвtrace_0
,:*2prediction_output_0/kernel
&:$2prediction_output_0/bias
.
G0
H1"
trackable_list_wrapper
ж
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
17"
trackable_list_wrapper
(
г0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
МBЙ
-__inference_joint_model_layer_call_fn_4114966	input_143	input_144"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
КBЗ
-__inference_joint_model_layer_call_fn_4115570inputs/0inputs/1"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
КBЗ
-__inference_joint_model_layer_call_fn_4115624inputs/0inputs/1"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
МBЙ
-__inference_joint_model_layer_call_fn_4115316	input_143	input_144"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
еBв
H__inference_joint_model_layer_call_and_return_conditional_losses_4115721inputs/0inputs/1"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
еBв
H__inference_joint_model_layer_call_and_return_conditional_losses_4115836inputs/0inputs/1"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
зBд
H__inference_joint_model_layer_call_and_return_conditional_losses_4115385	input_143	input_144"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
зBд
H__inference_joint_model_layer_call_and_return_conditional_losses_4115454	input_143	input_144"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
╫B╘
%__inference_signature_wrapper_4115516	input_143	input_144"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
рB▌
,__inference_conv2d_429_layer_call_fn_4115845inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_conv2d_429_layer_call_and_return_conditional_losses_4115856inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
√B°
6__inference_spatial_dropout2d_71_layer_call_fn_4115861inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
6__inference_spatial_dropout2d_71_layer_call_fn_4115866inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЦBУ
Q__inference_spatial_dropout2d_71_layer_call_and_return_conditional_losses_4115871inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЦBУ
Q__inference_spatial_dropout2d_71_layer_call_and_return_conditional_losses_4115894inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
рB▌
,__inference_conv2d_426_layer_call_fn_4115903inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_conv2d_426_layer_call_and_return_conditional_losses_4115914inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
рB▌
,__inference_conv2d_430_layer_call_fn_4115923inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_conv2d_430_layer_call_and_return_conditional_losses_4115934inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
¤B·
8__inference_batch_normalization_71_layer_call_fn_4115947inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
8__inference_batch_normalization_71_layer_call_fn_4115960inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ШBХ
S__inference_batch_normalization_71_layer_call_and_return_conditional_losses_4115978inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ШBХ
S__inference_batch_normalization_71_layer_call_and_return_conditional_losses_4115996inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
рB▌
,__inference_conv2d_431_layer_call_fn_4116005inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_conv2d_431_layer_call_and_return_conditional_losses_4116016inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
рB▌
,__inference_conv2d_427_layer_call_fn_4116025inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_conv2d_427_layer_call_and_return_conditional_losses_4116036inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
эBъ
9__inference_global_max_pooling2d_71_layer_call_fn_4116041inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ИBЕ
T__inference_global_max_pooling2d_71_layer_call_and_return_conditional_losses_4116047inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
рB▌
,__inference_conv2d_428_layer_call_fn_4116056inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_conv2d_428_layer_call_and_return_conditional_losses_4116067inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
сB▐
-__inference_flatten_143_layer_call_fn_4116072inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
H__inference_flatten_143_layer_call_and_return_conditional_losses_4116078inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
сB▐
-__inference_flatten_142_layer_call_fn_4116083inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
H__inference_flatten_142_layer_call_and_return_conditional_losses_4116089inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
ЁBэ
0__inference_concatenate_71_layer_call_fn_4116095inputs/0inputs/1"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЛBИ
K__inference_concatenate_71_layer_call_and_return_conditional_losses_4116102inputs/0inputs/1"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
+__inference_dense_213_layer_call_fn_4116111inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
F__inference_dense_213_layer_call_and_return_conditional_losses_4116122inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
+__inference_dense_214_layer_call_fn_4116131inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
F__inference_dense_214_layer_call_and_return_conditional_losses_4116142inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
+__inference_dense_215_layer_call_fn_4116151inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
F__inference_dense_215_layer_call_and_return_conditional_losses_4116162inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
щBц
5__inference_prediction_output_0_layer_call_fn_4116171inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ДBБ
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4116181inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
R
д	variables
е	keras_api

жtotal

зcount"
_tf_keras_metric
0
ж0
з1"
trackable_list_wrapper
.
д	variables"
_generic_user_object
:  (2total
:  (2count
0:.		2Adam/conv2d_429/kernel/m
": 2Adam/conv2d_429/bias/m
0:.2Adam/conv2d_426/kernel/m
": 2Adam/conv2d_426/bias/m
0:.2Adam/conv2d_430/kernel/m
": 2Adam/conv2d_430/bias/m
/:-2#Adam/batch_normalization_71/gamma/m
.:,2"Adam/batch_normalization_71/beta/m
0:.2Adam/conv2d_431/kernel/m
": 2Adam/conv2d_431/bias/m
0:.	2Adam/conv2d_427/kernel/m
": 2Adam/conv2d_427/bias/m
0:.2Adam/conv2d_428/kernel/m
": 2Adam/conv2d_428/bias/m
(:&	Ф2Adam/dense_213/kernel/m
!:2Adam/dense_213/bias/m
':%2Adam/dense_214/kernel/m
!:2Adam/dense_214/bias/m
':%2Adam/dense_215/kernel/m
!:2Adam/dense_215/bias/m
1:/2!Adam/prediction_output_0/kernel/m
+:)2Adam/prediction_output_0/bias/m
0:.		2Adam/conv2d_429/kernel/v
": 2Adam/conv2d_429/bias/v
0:.2Adam/conv2d_426/kernel/v
": 2Adam/conv2d_426/bias/v
0:.2Adam/conv2d_430/kernel/v
": 2Adam/conv2d_430/bias/v
/:-2#Adam/batch_normalization_71/gamma/v
.:,2"Adam/batch_normalization_71/beta/v
0:.2Adam/conv2d_431/kernel/v
": 2Adam/conv2d_431/bias/v
0:.	2Adam/conv2d_427/kernel/v
": 2Adam/conv2d_427/bias/v
0:.2Adam/conv2d_428/kernel/v
": 2Adam/conv2d_428/bias/v
(:&	Ф2Adam/dense_213/kernel/v
!:2Adam/dense_213/bias/v
':%2Adam/dense_214/kernel/v
!:2Adam/dense_214/bias/v
':%2Adam/dense_215/kernel/v
!:2Adam/dense_215/bias/v
1:/2!Adam/prediction_output_0/kernel/v
+:)2Adam/prediction_output_0/bias/vЕ
"__inference__wrapped_model_4114583▐ "#23EFGH;<XYOPghВГКЛТУЪЫoвl
eвb
`Ъ]
,К)
	input_143         Й
-К*
	input_144         ЙЙ
к "IкF
D
prediction_output_0-К*
prediction_output_0         ю
S__inference_batch_normalization_71_layer_call_and_return_conditional_losses_4115978ЦEFGHMвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ ю
S__inference_batch_normalization_71_layer_call_and_return_conditional_losses_4115996ЦEFGHMвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ ╞
8__inference_batch_normalization_71_layer_call_fn_4115947ЙEFGHMвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           ╞
8__inference_batch_normalization_71_layer_call_fn_4115960ЙEFGHMвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           ╒
K__inference_concatenate_71_layer_call_and_return_conditional_losses_4116102Е[вX
QвN
LЪI
"К
inputs/0         
#К 
inputs/1         Т
к "&в#
К
0         Ф
Ъ м
0__inference_concatenate_71_layer_call_fn_4116095x[вX
QвN
LЪI
"К
inputs/0         
#К 
inputs/1         Т
к "К         Ф╣
G__inference_conv2d_426_layer_call_and_return_conditional_losses_4115914n238в5
.в+
)К&
inputs         Й
к ".в+
$К!
0         Й
Ъ С
,__inference_conv2d_426_layer_call_fn_4115903a238в5
.в+
)К&
inputs         Й
к "!К         Й╣
G__inference_conv2d_427_layer_call_and_return_conditional_losses_4116036nXY8в5
.в+
)К&
inputs         Й
к ".в+
$К!
0         Й
Ъ С
,__inference_conv2d_427_layer_call_fn_4116025aXY8в5
.в+
)К&
inputs         Й
к "!К         Й╣
G__inference_conv2d_428_layer_call_and_return_conditional_losses_4116067ngh8в5
.в+
)К&
inputs         Й
к ".в+
$К!
0         Й
Ъ С
,__inference_conv2d_428_layer_call_fn_4116056agh8в5
.в+
)К&
inputs         Й
к "!К         Й║
G__inference_conv2d_429_layer_call_and_return_conditional_losses_4115856o"#9в6
/в,
*К'
inputs         ЙЙ
к ".в+
$К!
0         Й
Ъ Т
,__inference_conv2d_429_layer_call_fn_4115845b"#9в6
/в,
*К'
inputs         ЙЙ
к "!К         Й╣
G__inference_conv2d_430_layer_call_and_return_conditional_losses_4115934n;<8в5
.в+
)К&
inputs         Й
к ".в+
$К!
0         Й
Ъ С
,__inference_conv2d_430_layer_call_fn_4115923a;<8в5
.в+
)К&
inputs         Й
к "!К         Й╣
G__inference_conv2d_431_layer_call_and_return_conditional_losses_4116016nOP8в5
.в+
)К&
inputs         Й
к ".в+
$К!
0         Й
Ъ С
,__inference_conv2d_431_layer_call_fn_4116005aOP8в5
.в+
)К&
inputs         Й
к "!К         Йй
F__inference_dense_213_layer_call_and_return_conditional_losses_4116122_ВГ0в-
&в#
!К
inputs         Ф
к "%в"
К
0         
Ъ Б
+__inference_dense_213_layer_call_fn_4116111RВГ0в-
&в#
!К
inputs         Ф
к "К         и
F__inference_dense_214_layer_call_and_return_conditional_losses_4116142^КЛ/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ А
+__inference_dense_214_layer_call_fn_4116131QКЛ/в,
%в"
 К
inputs         
к "К         и
F__inference_dense_215_layer_call_and_return_conditional_losses_4116162^ТУ/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ А
+__inference_dense_215_layer_call_fn_4116151QТУ/в,
%в"
 К
inputs         
к "К         о
H__inference_flatten_142_layer_call_and_return_conditional_losses_4116089b8в5
.в+
)К&
inputs         Й
к "&в#
К
0         Т
Ъ Ж
-__inference_flatten_142_layer_call_fn_4116083U8в5
.в+
)К&
inputs         Й
к "К         Тд
H__inference_flatten_143_layer_call_and_return_conditional_losses_4116078X/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ |
-__inference_flatten_143_layer_call_fn_4116072K/в,
%в"
 К
inputs         
к "К         ▌
T__inference_global_max_pooling2d_71_layer_call_and_return_conditional_losses_4116047ДRвO
HвE
CК@
inputs4                                    
к ".в+
$К!
0                  
Ъ ┤
9__inference_global_max_pooling2d_71_layer_call_fn_4116041wRвO
HвE
CК@
inputs4                                    
к "!К                  П
H__inference_joint_model_layer_call_and_return_conditional_losses_4115385┬ "#23EFGH;<XYOPghВГКЛТУЪЫwвt
mвj
`Ъ]
,К)
	input_143         Й
-К*
	input_144         ЙЙ
p 

 
к "%в"
К
0         
Ъ П
H__inference_joint_model_layer_call_and_return_conditional_losses_4115454┬ "#23EFGH;<XYOPghВГКЛТУЪЫwвt
mвj
`Ъ]
,К)
	input_143         Й
-К*
	input_144         ЙЙ
p

 
к "%в"
К
0         
Ъ Н
H__inference_joint_model_layer_call_and_return_conditional_losses_4115721└ "#23EFGH;<XYOPghВГКЛТУЪЫuвr
kвh
^Ъ[
+К(
inputs/0         Й
,К)
inputs/1         ЙЙ
p 

 
к "%в"
К
0         
Ъ Н
H__inference_joint_model_layer_call_and_return_conditional_losses_4115836└ "#23EFGH;<XYOPghВГКЛТУЪЫuвr
kвh
^Ъ[
+К(
inputs/0         Й
,К)
inputs/1         ЙЙ
p

 
к "%в"
К
0         
Ъ ч
-__inference_joint_model_layer_call_fn_4114966╡ "#23EFGH;<XYOPghВГКЛТУЪЫwвt
mвj
`Ъ]
,К)
	input_143         Й
-К*
	input_144         ЙЙ
p 

 
к "К         ч
-__inference_joint_model_layer_call_fn_4115316╡ "#23EFGH;<XYOPghВГКЛТУЪЫwвt
mвj
`Ъ]
,К)
	input_143         Й
-К*
	input_144         ЙЙ
p

 
к "К         х
-__inference_joint_model_layer_call_fn_4115570│ "#23EFGH;<XYOPghВГКЛТУЪЫuвr
kвh
^Ъ[
+К(
inputs/0         Й
,К)
inputs/1         ЙЙ
p 

 
к "К         х
-__inference_joint_model_layer_call_fn_4115624│ "#23EFGH;<XYOPghВГКЛТУЪЫuвr
kвh
^Ъ[
+К(
inputs/0         Й
,К)
inputs/1         ЙЙ
p

 
к "К         ▓
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4116181^ЪЫ/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ К
5__inference_prediction_output_0_layer_call_fn_4116171QЪЫ/в,
%в"
 К
inputs         
к "К         Я
%__inference_signature_wrapper_4115516ї "#23EFGH;<XYOPghВГКЛТУЪЫЕвБ
в 
zкw
9
	input_143,К)
	input_143         Й
:
	input_144-К*
	input_144         ЙЙ"IкF
D
prediction_output_0-К*
prediction_output_0         °
Q__inference_spatial_dropout2d_71_layer_call_and_return_conditional_losses_4115871вVвS
LвI
CК@
inputs4                                    
p 
к "HвE
>К;
04                                    
Ъ °
Q__inference_spatial_dropout2d_71_layer_call_and_return_conditional_losses_4115894вVвS
LвI
CК@
inputs4                                    
p
к "HвE
>К;
04                                    
Ъ ╨
6__inference_spatial_dropout2d_71_layer_call_fn_4115861ХVвS
LвI
CК@
inputs4                                    
p 
к ";К84                                    ╨
6__inference_spatial_dropout2d_71_layer_call_fn_4115866ХVвS
LвI
CК@
inputs4                                    
p
к ";К84                                    
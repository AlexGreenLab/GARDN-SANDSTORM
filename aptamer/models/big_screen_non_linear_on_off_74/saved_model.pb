ЁЦ
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
 И"serve*2.10.02unknown8╘є
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
Adam/dense_329/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_329/bias/v
{
)Adam/dense_329/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_329/bias/v*
_output_shapes
:*
dtype0
К
Adam/dense_329/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_329/kernel/v
Г
+Adam/dense_329/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_329/kernel/v*
_output_shapes

:*
dtype0
В
Adam/dense_328/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_328/bias/v
{
)Adam/dense_328/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_328/bias/v*
_output_shapes
:*
dtype0
К
Adam/dense_328/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_328/kernel/v
Г
+Adam/dense_328/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_328/kernel/v*
_output_shapes

:*
dtype0
В
Adam/dense_327/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_327/bias/v
{
)Adam/dense_327/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_327/bias/v*
_output_shapes
:*
dtype0
Л
Adam/dense_327/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ф*(
shared_nameAdam/dense_327/kernel/v
Д
+Adam/dense_327/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_327/kernel/v*
_output_shapes
:	Ф*
dtype0
Д
Adam/conv2d_656/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_656/bias/v
}
*Adam/conv2d_656/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_656/bias/v*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_656/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_656/kernel/v
Н
,Adam/conv2d_656/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_656/kernel/v*&
_output_shapes
:*
dtype0
Д
Adam/conv2d_655/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_655/bias/v
}
*Adam/conv2d_655/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_655/bias/v*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_655/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/conv2d_655/kernel/v
Н
,Adam/conv2d_655/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_655/kernel/v*&
_output_shapes
:	*
dtype0
Д
Adam/conv2d_659/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_659/bias/v
}
*Adam/conv2d_659/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_659/bias/v*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_659/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_659/kernel/v
Н
,Adam/conv2d_659/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_659/kernel/v*&
_output_shapes
:*
dtype0
Ю
#Adam/batch_normalization_109/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_109/beta/v
Ч
7Adam/batch_normalization_109/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_109/beta/v*
_output_shapes
:*
dtype0
а
$Adam/batch_normalization_109/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_109/gamma/v
Щ
8Adam/batch_normalization_109/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_109/gamma/v*
_output_shapes
:*
dtype0
Д
Adam/conv2d_658/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_658/bias/v
}
*Adam/conv2d_658/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_658/bias/v*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_658/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_658/kernel/v
Н
,Adam/conv2d_658/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_658/kernel/v*&
_output_shapes
:*
dtype0
Д
Adam/conv2d_654/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_654/bias/v
}
*Adam/conv2d_654/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_654/bias/v*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_654/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_654/kernel/v
Н
,Adam/conv2d_654/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_654/kernel/v*&
_output_shapes
:*
dtype0
Д
Adam/conv2d_657/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_657/bias/v
}
*Adam/conv2d_657/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_657/bias/v*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_657/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*)
shared_nameAdam/conv2d_657/kernel/v
Н
,Adam/conv2d_657/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_657/kernel/v*&
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
Adam/dense_329/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_329/bias/m
{
)Adam/dense_329/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_329/bias/m*
_output_shapes
:*
dtype0
К
Adam/dense_329/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_329/kernel/m
Г
+Adam/dense_329/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_329/kernel/m*
_output_shapes

:*
dtype0
В
Adam/dense_328/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_328/bias/m
{
)Adam/dense_328/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_328/bias/m*
_output_shapes
:*
dtype0
К
Adam/dense_328/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_328/kernel/m
Г
+Adam/dense_328/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_328/kernel/m*
_output_shapes

:*
dtype0
В
Adam/dense_327/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_327/bias/m
{
)Adam/dense_327/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_327/bias/m*
_output_shapes
:*
dtype0
Л
Adam/dense_327/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ф*(
shared_nameAdam/dense_327/kernel/m
Д
+Adam/dense_327/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_327/kernel/m*
_output_shapes
:	Ф*
dtype0
Д
Adam/conv2d_656/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_656/bias/m
}
*Adam/conv2d_656/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_656/bias/m*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_656/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_656/kernel/m
Н
,Adam/conv2d_656/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_656/kernel/m*&
_output_shapes
:*
dtype0
Д
Adam/conv2d_655/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_655/bias/m
}
*Adam/conv2d_655/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_655/bias/m*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_655/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/conv2d_655/kernel/m
Н
,Adam/conv2d_655/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_655/kernel/m*&
_output_shapes
:	*
dtype0
Д
Adam/conv2d_659/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_659/bias/m
}
*Adam/conv2d_659/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_659/bias/m*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_659/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_659/kernel/m
Н
,Adam/conv2d_659/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_659/kernel/m*&
_output_shapes
:*
dtype0
Ю
#Adam/batch_normalization_109/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_109/beta/m
Ч
7Adam/batch_normalization_109/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_109/beta/m*
_output_shapes
:*
dtype0
а
$Adam/batch_normalization_109/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_109/gamma/m
Щ
8Adam/batch_normalization_109/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_109/gamma/m*
_output_shapes
:*
dtype0
Д
Adam/conv2d_658/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_658/bias/m
}
*Adam/conv2d_658/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_658/bias/m*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_658/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_658/kernel/m
Н
,Adam/conv2d_658/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_658/kernel/m*&
_output_shapes
:*
dtype0
Д
Adam/conv2d_654/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_654/bias/m
}
*Adam/conv2d_654/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_654/bias/m*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_654/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_654/kernel/m
Н
,Adam/conv2d_654/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_654/kernel/m*&
_output_shapes
:*
dtype0
Д
Adam/conv2d_657/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_657/bias/m
}
*Adam/conv2d_657/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_657/bias/m*
_output_shapes
:*
dtype0
Ф
Adam/conv2d_657/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*)
shared_nameAdam/conv2d_657/kernel/m
Н
,Adam/conv2d_657/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_657/kernel/m*&
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
dense_329/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_329/bias
m
"dense_329/bias/Read/ReadVariableOpReadVariableOpdense_329/bias*
_output_shapes
:*
dtype0
|
dense_329/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_329/kernel
u
$dense_329/kernel/Read/ReadVariableOpReadVariableOpdense_329/kernel*
_output_shapes

:*
dtype0
t
dense_328/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_328/bias
m
"dense_328/bias/Read/ReadVariableOpReadVariableOpdense_328/bias*
_output_shapes
:*
dtype0
|
dense_328/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_328/kernel
u
$dense_328/kernel/Read/ReadVariableOpReadVariableOpdense_328/kernel*
_output_shapes

:*
dtype0
t
dense_327/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_327/bias
m
"dense_327/bias/Read/ReadVariableOpReadVariableOpdense_327/bias*
_output_shapes
:*
dtype0
}
dense_327/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ф*!
shared_namedense_327/kernel
v
$dense_327/kernel/Read/ReadVariableOpReadVariableOpdense_327/kernel*
_output_shapes
:	Ф*
dtype0
v
conv2d_656/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_656/bias
o
#conv2d_656/bias/Read/ReadVariableOpReadVariableOpconv2d_656/bias*
_output_shapes
:*
dtype0
Ж
conv2d_656/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_656/kernel

%conv2d_656/kernel/Read/ReadVariableOpReadVariableOpconv2d_656/kernel*&
_output_shapes
:*
dtype0
v
conv2d_655/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_655/bias
o
#conv2d_655/bias/Read/ReadVariableOpReadVariableOpconv2d_655/bias*
_output_shapes
:*
dtype0
Ж
conv2d_655/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*"
shared_nameconv2d_655/kernel

%conv2d_655/kernel/Read/ReadVariableOpReadVariableOpconv2d_655/kernel*&
_output_shapes
:	*
dtype0
v
conv2d_659/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_659/bias
o
#conv2d_659/bias/Read/ReadVariableOpReadVariableOpconv2d_659/bias*
_output_shapes
:*
dtype0
Ж
conv2d_659/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_659/kernel

%conv2d_659/kernel/Read/ReadVariableOpReadVariableOpconv2d_659/kernel*&
_output_shapes
:*
dtype0
ж
'batch_normalization_109/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_109/moving_variance
Я
;batch_normalization_109/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_109/moving_variance*
_output_shapes
:*
dtype0
Ю
#batch_normalization_109/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_109/moving_mean
Ч
7batch_normalization_109/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_109/moving_mean*
_output_shapes
:*
dtype0
Р
batch_normalization_109/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_109/beta
Й
0batch_normalization_109/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_109/beta*
_output_shapes
:*
dtype0
Т
batch_normalization_109/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_109/gamma
Л
1batch_normalization_109/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_109/gamma*
_output_shapes
:*
dtype0
v
conv2d_658/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_658/bias
o
#conv2d_658/bias/Read/ReadVariableOpReadVariableOpconv2d_658/bias*
_output_shapes
:*
dtype0
Ж
conv2d_658/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_658/kernel

%conv2d_658/kernel/Read/ReadVariableOpReadVariableOpconv2d_658/kernel*&
_output_shapes
:*
dtype0
v
conv2d_654/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_654/bias
o
#conv2d_654/bias/Read/ReadVariableOpReadVariableOpconv2d_654/bias*
_output_shapes
:*
dtype0
Ж
conv2d_654/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_654/kernel

%conv2d_654/kernel/Read/ReadVariableOpReadVariableOpconv2d_654/kernel*&
_output_shapes
:*
dtype0
v
conv2d_657/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_657/bias
o
#conv2d_657/bias/Read/ReadVariableOpReadVariableOpconv2d_657/bias*
_output_shapes
:*
dtype0
Ж
conv2d_657/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*"
shared_nameconv2d_657/kernel

%conv2d_657/kernel/Read/ReadVariableOpReadVariableOpconv2d_657/kernel*&
_output_shapes
:		*
dtype0
О
serving_default_input_219Placeholder*0
_output_shapes
:         Й*
dtype0*%
shape:         Й
Р
serving_default_input_220Placeholder*1
_output_shapes
:         ЙЙ*
dtype0*&
shape:         ЙЙ
Г
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_219serving_default_input_220conv2d_657/kernelconv2d_657/biasconv2d_654/kernelconv2d_654/biasbatch_normalization_109/gammabatch_normalization_109/beta#batch_normalization_109/moving_mean'batch_normalization_109/moving_varianceconv2d_658/kernelconv2d_658/biasconv2d_655/kernelconv2d_655/biasconv2d_659/kernelconv2d_659/biasconv2d_656/kernelconv2d_656/biasdense_327/kerneldense_327/biasdense_328/kerneldense_328/biasdense_329/kerneldense_329/biasprediction_output_0/kernelprediction_output_0/bias*%
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
%__inference_signature_wrapper_4210554

NoOpNoOp
№а
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╢а
valueлаBза BЯа
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
VARIABLE_VALUEconv2d_657/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_657/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_654/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_654/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_658/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_658/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
lf
VARIABLE_VALUEbatch_normalization_109/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_109/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_109/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUE'batch_normalization_109/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_659/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_659/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_655/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_655/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_656/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_656/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_327/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_327/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_328/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_328/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_329/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_329/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/conv2d_657/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_657/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_654/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_654/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_658/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_658/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
РЙ
VARIABLE_VALUE$Adam/batch_normalization_109/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE#Adam/batch_normalization_109/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_659/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_659/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_655/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_655/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_656/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_656/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdam/dense_327/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_327/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdam/dense_328/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_328/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdam/dense_329/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_329/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE!Adam/prediction_output_0/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЛД
VARIABLE_VALUEAdam/prediction_output_0/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_657/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_657/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_654/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_654/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_658/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_658/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
РЙ
VARIABLE_VALUE$Adam/batch_normalization_109/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE#Adam/batch_normalization_109/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_659/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_659/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_655/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_655/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/conv2d_656/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d_656/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdam/dense_327/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_327/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdam/dense_328/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_328/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdam/dense_329/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_329/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
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
Ъ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%conv2d_657/kernel/Read/ReadVariableOp#conv2d_657/bias/Read/ReadVariableOp%conv2d_654/kernel/Read/ReadVariableOp#conv2d_654/bias/Read/ReadVariableOp%conv2d_658/kernel/Read/ReadVariableOp#conv2d_658/bias/Read/ReadVariableOp1batch_normalization_109/gamma/Read/ReadVariableOp0batch_normalization_109/beta/Read/ReadVariableOp7batch_normalization_109/moving_mean/Read/ReadVariableOp;batch_normalization_109/moving_variance/Read/ReadVariableOp%conv2d_659/kernel/Read/ReadVariableOp#conv2d_659/bias/Read/ReadVariableOp%conv2d_655/kernel/Read/ReadVariableOp#conv2d_655/bias/Read/ReadVariableOp%conv2d_656/kernel/Read/ReadVariableOp#conv2d_656/bias/Read/ReadVariableOp$dense_327/kernel/Read/ReadVariableOp"dense_327/bias/Read/ReadVariableOp$dense_328/kernel/Read/ReadVariableOp"dense_328/bias/Read/ReadVariableOp$dense_329/kernel/Read/ReadVariableOp"dense_329/bias/Read/ReadVariableOp.prediction_output_0/kernel/Read/ReadVariableOp,prediction_output_0/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp,Adam/conv2d_657/kernel/m/Read/ReadVariableOp*Adam/conv2d_657/bias/m/Read/ReadVariableOp,Adam/conv2d_654/kernel/m/Read/ReadVariableOp*Adam/conv2d_654/bias/m/Read/ReadVariableOp,Adam/conv2d_658/kernel/m/Read/ReadVariableOp*Adam/conv2d_658/bias/m/Read/ReadVariableOp8Adam/batch_normalization_109/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_109/beta/m/Read/ReadVariableOp,Adam/conv2d_659/kernel/m/Read/ReadVariableOp*Adam/conv2d_659/bias/m/Read/ReadVariableOp,Adam/conv2d_655/kernel/m/Read/ReadVariableOp*Adam/conv2d_655/bias/m/Read/ReadVariableOp,Adam/conv2d_656/kernel/m/Read/ReadVariableOp*Adam/conv2d_656/bias/m/Read/ReadVariableOp+Adam/dense_327/kernel/m/Read/ReadVariableOp)Adam/dense_327/bias/m/Read/ReadVariableOp+Adam/dense_328/kernel/m/Read/ReadVariableOp)Adam/dense_328/bias/m/Read/ReadVariableOp+Adam/dense_329/kernel/m/Read/ReadVariableOp)Adam/dense_329/bias/m/Read/ReadVariableOp5Adam/prediction_output_0/kernel/m/Read/ReadVariableOp3Adam/prediction_output_0/bias/m/Read/ReadVariableOp,Adam/conv2d_657/kernel/v/Read/ReadVariableOp*Adam/conv2d_657/bias/v/Read/ReadVariableOp,Adam/conv2d_654/kernel/v/Read/ReadVariableOp*Adam/conv2d_654/bias/v/Read/ReadVariableOp,Adam/conv2d_658/kernel/v/Read/ReadVariableOp*Adam/conv2d_658/bias/v/Read/ReadVariableOp8Adam/batch_normalization_109/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_109/beta/v/Read/ReadVariableOp,Adam/conv2d_659/kernel/v/Read/ReadVariableOp*Adam/conv2d_659/bias/v/Read/ReadVariableOp,Adam/conv2d_655/kernel/v/Read/ReadVariableOp*Adam/conv2d_655/bias/v/Read/ReadVariableOp,Adam/conv2d_656/kernel/v/Read/ReadVariableOp*Adam/conv2d_656/bias/v/Read/ReadVariableOp+Adam/dense_327/kernel/v/Read/ReadVariableOp)Adam/dense_327/bias/v/Read/ReadVariableOp+Adam/dense_328/kernel/v/Read/ReadVariableOp)Adam/dense_328/bias/v/Read/ReadVariableOp+Adam/dense_329/kernel/v/Read/ReadVariableOp)Adam/dense_329/bias/v/Read/ReadVariableOp5Adam/prediction_output_0/kernel/v/Read/ReadVariableOp3Adam/prediction_output_0/bias/v/Read/ReadVariableOpConst*X
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
 __inference__traced_save_4211468
╣
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_657/kernelconv2d_657/biasconv2d_654/kernelconv2d_654/biasconv2d_658/kernelconv2d_658/biasbatch_normalization_109/gammabatch_normalization_109/beta#batch_normalization_109/moving_mean'batch_normalization_109/moving_varianceconv2d_659/kernelconv2d_659/biasconv2d_655/kernelconv2d_655/biasconv2d_656/kernelconv2d_656/biasdense_327/kerneldense_327/biasdense_328/kerneldense_328/biasdense_329/kerneldense_329/biasprediction_output_0/kernelprediction_output_0/biasbeta_1beta_2decaylearning_rate	Adam/itertotalcountAdam/conv2d_657/kernel/mAdam/conv2d_657/bias/mAdam/conv2d_654/kernel/mAdam/conv2d_654/bias/mAdam/conv2d_658/kernel/mAdam/conv2d_658/bias/m$Adam/batch_normalization_109/gamma/m#Adam/batch_normalization_109/beta/mAdam/conv2d_659/kernel/mAdam/conv2d_659/bias/mAdam/conv2d_655/kernel/mAdam/conv2d_655/bias/mAdam/conv2d_656/kernel/mAdam/conv2d_656/bias/mAdam/dense_327/kernel/mAdam/dense_327/bias/mAdam/dense_328/kernel/mAdam/dense_328/bias/mAdam/dense_329/kernel/mAdam/dense_329/bias/m!Adam/prediction_output_0/kernel/mAdam/prediction_output_0/bias/mAdam/conv2d_657/kernel/vAdam/conv2d_657/bias/vAdam/conv2d_654/kernel/vAdam/conv2d_654/bias/vAdam/conv2d_658/kernel/vAdam/conv2d_658/bias/v$Adam/batch_normalization_109/gamma/v#Adam/batch_normalization_109/beta/vAdam/conv2d_659/kernel/vAdam/conv2d_659/bias/vAdam/conv2d_655/kernel/vAdam/conv2d_655/bias/vAdam/conv2d_656/kernel/vAdam/conv2d_656/bias/vAdam/dense_327/kernel/vAdam/dense_327/bias/vAdam/dense_328/kernel/vAdam/dense_328/bias/vAdam/dense_329/kernel/vAdam/dense_329/bias/v!Adam/prediction_output_0/kernel/vAdam/prediction_output_0/bias/v*W
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
#__inference__traced_restore_4211703КА
╔
Ш
+__inference_dense_328_layer_call_fn_4211169

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
F__inference_dense_328_layer_call_and_return_conditional_losses_4209913o
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
╝
I
-__inference_flatten_218_layer_call_fn_4211121

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
H__inference_flatten_218_layer_call_and_return_conditional_losses_4209874a
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
ё
p
R__inference_spatial_dropout2d_109_layer_call_and_return_conditional_losses_4210909

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
∙
б
,__inference_conv2d_657_layer_call_fn_4210883

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
G__inference_conv2d_657_layer_call_and_return_conditional_losses_4209758x
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
ЁN
ь
H__inference_joint_model_layer_call_and_return_conditional_losses_4210423
	input_219
	input_220,
conv2d_657_4210358:		 
conv2d_657_4210360:,
conv2d_654_4210363: 
conv2d_654_4210365:-
batch_normalization_109_4210369:-
batch_normalization_109_4210371:-
batch_normalization_109_4210373:-
batch_normalization_109_4210375:,
conv2d_658_4210378: 
conv2d_658_4210380:,
conv2d_655_4210383:	 
conv2d_655_4210385:,
conv2d_659_4210388: 
conv2d_659_4210390:,
conv2d_656_4210393: 
conv2d_656_4210395:$
dense_327_4210402:	Ф
dense_327_4210404:#
dense_328_4210407:
dense_328_4210409:#
dense_329_4210412:
dense_329_4210414:-
prediction_output_0_4210417:)
prediction_output_0_4210419:
identityИв/batch_normalization_109/StatefulPartitionedCallв"conv2d_654/StatefulPartitionedCallв"conv2d_655/StatefulPartitionedCallв"conv2d_656/StatefulPartitionedCallв"conv2d_657/StatefulPartitionedCallв"conv2d_658/StatefulPartitionedCallв"conv2d_659/StatefulPartitionedCallв!dense_327/StatefulPartitionedCallв!dense_328/StatefulPartitionedCallв!dense_329/StatefulPartitionedCallв+prediction_output_0/StatefulPartitionedCallК
"conv2d_657/StatefulPartitionedCallStatefulPartitionedCall	input_220conv2d_657_4210358conv2d_657_4210360*
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
G__inference_conv2d_657_layer_call_and_return_conditional_losses_4209758К
"conv2d_654/StatefulPartitionedCallStatefulPartitionedCall	input_219conv2d_654_4210363conv2d_654_4210365*
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
G__inference_conv2d_654_layer_call_and_return_conditional_losses_4209775Д
%spatial_dropout2d_109/PartitionedCallPartitionedCall+conv2d_657/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *[
fVRT
R__inference_spatial_dropout2d_109_layer_call_and_return_conditional_losses_4209630ж
/batch_normalization_109/StatefulPartitionedCallStatefulPartitionedCall+conv2d_654/StatefulPartitionedCall:output:0batch_normalization_109_4210369batch_normalization_109_4210371batch_normalization_109_4210373batch_normalization_109_4210375*
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
GPU2*0J 8В *]
fXRV
T__inference_batch_normalization_109_layer_call_and_return_conditional_losses_4209683п
"conv2d_658/StatefulPartitionedCallStatefulPartitionedCall.spatial_dropout2d_109/PartitionedCall:output:0conv2d_658_4210378conv2d_658_4210380*
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
G__inference_conv2d_658_layer_call_and_return_conditional_losses_4209802╣
"conv2d_655/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_109/StatefulPartitionedCall:output:0conv2d_655_4210383conv2d_655_4210385*
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
G__inference_conv2d_655_layer_call_and_return_conditional_losses_4209819м
"conv2d_659/StatefulPartitionedCallStatefulPartitionedCall+conv2d_658/StatefulPartitionedCall:output:0conv2d_659_4210388conv2d_659_4210390*
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
G__inference_conv2d_659_layer_call_and_return_conditional_losses_4209836м
"conv2d_656/StatefulPartitionedCallStatefulPartitionedCall+conv2d_655/StatefulPartitionedCall:output:0conv2d_656_4210393conv2d_656_4210395*
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
G__inference_conv2d_656_layer_call_and_return_conditional_losses_4209853Б
(global_max_pooling2d_109/PartitionedCallPartitionedCall+conv2d_659/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *^
fYRW
U__inference_global_max_pooling2d_109_layer_call_and_return_conditional_losses_4209735э
flatten_219/PartitionedCallPartitionedCall1global_max_pooling2d_109/PartitionedCall:output:0*
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
H__inference_flatten_219_layer_call_and_return_conditional_losses_4209866ш
flatten_218/PartitionedCallPartitionedCall+conv2d_656/StatefulPartitionedCall:output:0*
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
H__inference_flatten_218_layer_call_and_return_conditional_losses_4209874Р
concatenate_109/PartitionedCallPartitionedCall$flatten_219/PartitionedCall:output:0$flatten_218/PartitionedCall:output:0*
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
GPU2*0J 8В *U
fPRN
L__inference_concatenate_109_layer_call_and_return_conditional_losses_4209883Ь
!dense_327/StatefulPartitionedCallStatefulPartitionedCall(concatenate_109/PartitionedCall:output:0dense_327_4210402dense_327_4210404*
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
F__inference_dense_327_layer_call_and_return_conditional_losses_4209896Ю
!dense_328/StatefulPartitionedCallStatefulPartitionedCall*dense_327/StatefulPartitionedCall:output:0dense_328_4210407dense_328_4210409*
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
F__inference_dense_328_layer_call_and_return_conditional_losses_4209913Ю
!dense_329/StatefulPartitionedCallStatefulPartitionedCall*dense_328/StatefulPartitionedCall:output:0dense_329_4210412dense_329_4210414*
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
F__inference_dense_329_layer_call_and_return_conditional_losses_4209930╞
+prediction_output_0/StatefulPartitionedCallStatefulPartitionedCall*dense_329/StatefulPartitionedCall:output:0prediction_output_0_4210417prediction_output_0_4210419*
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
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4209946Г
IdentityIdentity4prediction_output_0/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Ё
NoOpNoOp0^batch_normalization_109/StatefulPartitionedCall#^conv2d_654/StatefulPartitionedCall#^conv2d_655/StatefulPartitionedCall#^conv2d_656/StatefulPartitionedCall#^conv2d_657/StatefulPartitionedCall#^conv2d_658/StatefulPartitionedCall#^conv2d_659/StatefulPartitionedCall"^dense_327/StatefulPartitionedCall"^dense_328/StatefulPartitionedCall"^dense_329/StatefulPartitionedCall,^prediction_output_0/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:         Й:         ЙЙ: : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_109/StatefulPartitionedCall/batch_normalization_109/StatefulPartitionedCall2H
"conv2d_654/StatefulPartitionedCall"conv2d_654/StatefulPartitionedCall2H
"conv2d_655/StatefulPartitionedCall"conv2d_655/StatefulPartitionedCall2H
"conv2d_656/StatefulPartitionedCall"conv2d_656/StatefulPartitionedCall2H
"conv2d_657/StatefulPartitionedCall"conv2d_657/StatefulPartitionedCall2H
"conv2d_658/StatefulPartitionedCall"conv2d_658/StatefulPartitionedCall2H
"conv2d_659/StatefulPartitionedCall"conv2d_659/StatefulPartitionedCall2F
!dense_327/StatefulPartitionedCall!dense_327/StatefulPartitionedCall2F
!dense_328/StatefulPartitionedCall!dense_328/StatefulPartitionedCall2F
!dense_329/StatefulPartitionedCall!dense_329/StatefulPartitionedCall2Z
+prediction_output_0/StatefulPartitionedCall+prediction_output_0/StatefulPartitionedCall:[ W
0
_output_shapes
:         Й
#
_user_specified_name	input_219:\X
1
_output_shapes
:         ЙЙ
#
_user_specified_name	input_220
п
м
%__inference_signature_wrapper_4210554
	input_219
	input_220!
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
StatefulPartitionedCallStatefulPartitionedCall	input_219	input_220unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
"__inference__wrapped_model_4209621o
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
_user_specified_name	input_219:\X
1
_output_shapes
:         ЙЙ
#
_user_specified_name	input_220
╕
d
H__inference_flatten_219_layer_call_and_return_conditional_losses_4211116

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
ў
б
,__inference_conv2d_656_layer_call_fn_4211094

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
G__inference_conv2d_656_layer_call_and_return_conditional_losses_4209853x
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
╧
Я
T__inference_batch_normalization_109_layer_call_and_return_conditional_losses_4211016

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
и
I
-__inference_flatten_219_layer_call_fn_4211110

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
H__inference_flatten_219_layer_call_and_return_conditional_losses_4209866`
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
ў
б
,__inference_conv2d_659_layer_call_fn_4211043

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
G__inference_conv2d_659_layer_call_and_return_conditional_losses_4209836x
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
М
А
G__inference_conv2d_654_layer_call_and_return_conditional_losses_4209775

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
б

°
F__inference_dense_327_layer_call_and_return_conditional_losses_4211160

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
╙	
Б
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4211219

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
╣
]
1__inference_concatenate_109_layer_call_fn_4211133
inputs_0
inputs_1
identity╚
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
GPU2*0J 8В *U
fPRN
L__inference_concatenate_109_layer_call_and_return_conditional_losses_4209883a
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
╞P
Ь
H__inference_joint_model_layer_call_and_return_conditional_losses_4210492
	input_219
	input_220,
conv2d_657_4210427:		 
conv2d_657_4210429:,
conv2d_654_4210432: 
conv2d_654_4210434:-
batch_normalization_109_4210438:-
batch_normalization_109_4210440:-
batch_normalization_109_4210442:-
batch_normalization_109_4210444:,
conv2d_658_4210447: 
conv2d_658_4210449:,
conv2d_655_4210452:	 
conv2d_655_4210454:,
conv2d_659_4210457: 
conv2d_659_4210459:,
conv2d_656_4210462: 
conv2d_656_4210464:$
dense_327_4210471:	Ф
dense_327_4210473:#
dense_328_4210476:
dense_328_4210478:#
dense_329_4210481:
dense_329_4210483:-
prediction_output_0_4210486:)
prediction_output_0_4210488:
identityИв/batch_normalization_109/StatefulPartitionedCallв"conv2d_654/StatefulPartitionedCallв"conv2d_655/StatefulPartitionedCallв"conv2d_656/StatefulPartitionedCallв"conv2d_657/StatefulPartitionedCallв"conv2d_658/StatefulPartitionedCallв"conv2d_659/StatefulPartitionedCallв!dense_327/StatefulPartitionedCallв!dense_328/StatefulPartitionedCallв!dense_329/StatefulPartitionedCallв+prediction_output_0/StatefulPartitionedCallв-spatial_dropout2d_109/StatefulPartitionedCallК
"conv2d_657/StatefulPartitionedCallStatefulPartitionedCall	input_220conv2d_657_4210427conv2d_657_4210429*
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
G__inference_conv2d_657_layer_call_and_return_conditional_losses_4209758К
"conv2d_654/StatefulPartitionedCallStatefulPartitionedCall	input_219conv2d_654_4210432conv2d_654_4210434*
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
G__inference_conv2d_654_layer_call_and_return_conditional_losses_4209775Ф
-spatial_dropout2d_109/StatefulPartitionedCallStatefulPartitionedCall+conv2d_657/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *[
fVRT
R__inference_spatial_dropout2d_109_layer_call_and_return_conditional_losses_4209658д
/batch_normalization_109/StatefulPartitionedCallStatefulPartitionedCall+conv2d_654/StatefulPartitionedCall:output:0batch_normalization_109_4210438batch_normalization_109_4210440batch_normalization_109_4210442batch_normalization_109_4210444*
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
GPU2*0J 8В *]
fXRV
T__inference_batch_normalization_109_layer_call_and_return_conditional_losses_4209714╖
"conv2d_658/StatefulPartitionedCallStatefulPartitionedCall6spatial_dropout2d_109/StatefulPartitionedCall:output:0conv2d_658_4210447conv2d_658_4210449*
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
G__inference_conv2d_658_layer_call_and_return_conditional_losses_4209802╣
"conv2d_655/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_109/StatefulPartitionedCall:output:0conv2d_655_4210452conv2d_655_4210454*
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
G__inference_conv2d_655_layer_call_and_return_conditional_losses_4209819м
"conv2d_659/StatefulPartitionedCallStatefulPartitionedCall+conv2d_658/StatefulPartitionedCall:output:0conv2d_659_4210457conv2d_659_4210459*
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
G__inference_conv2d_659_layer_call_and_return_conditional_losses_4209836м
"conv2d_656/StatefulPartitionedCallStatefulPartitionedCall+conv2d_655/StatefulPartitionedCall:output:0conv2d_656_4210462conv2d_656_4210464*
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
G__inference_conv2d_656_layer_call_and_return_conditional_losses_4209853Б
(global_max_pooling2d_109/PartitionedCallPartitionedCall+conv2d_659/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *^
fYRW
U__inference_global_max_pooling2d_109_layer_call_and_return_conditional_losses_4209735э
flatten_219/PartitionedCallPartitionedCall1global_max_pooling2d_109/PartitionedCall:output:0*
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
H__inference_flatten_219_layer_call_and_return_conditional_losses_4209866ш
flatten_218/PartitionedCallPartitionedCall+conv2d_656/StatefulPartitionedCall:output:0*
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
H__inference_flatten_218_layer_call_and_return_conditional_losses_4209874Р
concatenate_109/PartitionedCallPartitionedCall$flatten_219/PartitionedCall:output:0$flatten_218/PartitionedCall:output:0*
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
GPU2*0J 8В *U
fPRN
L__inference_concatenate_109_layer_call_and_return_conditional_losses_4209883Ь
!dense_327/StatefulPartitionedCallStatefulPartitionedCall(concatenate_109/PartitionedCall:output:0dense_327_4210471dense_327_4210473*
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
F__inference_dense_327_layer_call_and_return_conditional_losses_4209896Ю
!dense_328/StatefulPartitionedCallStatefulPartitionedCall*dense_327/StatefulPartitionedCall:output:0dense_328_4210476dense_328_4210478*
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
F__inference_dense_328_layer_call_and_return_conditional_losses_4209913Ю
!dense_329/StatefulPartitionedCallStatefulPartitionedCall*dense_328/StatefulPartitionedCall:output:0dense_329_4210481dense_329_4210483*
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
F__inference_dense_329_layer_call_and_return_conditional_losses_4209930╞
+prediction_output_0/StatefulPartitionedCallStatefulPartitionedCall*dense_329/StatefulPartitionedCall:output:0prediction_output_0_4210486prediction_output_0_4210488*
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
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4209946Г
IdentityIdentity4prediction_output_0/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         а
NoOpNoOp0^batch_normalization_109/StatefulPartitionedCall#^conv2d_654/StatefulPartitionedCall#^conv2d_655/StatefulPartitionedCall#^conv2d_656/StatefulPartitionedCall#^conv2d_657/StatefulPartitionedCall#^conv2d_658/StatefulPartitionedCall#^conv2d_659/StatefulPartitionedCall"^dense_327/StatefulPartitionedCall"^dense_328/StatefulPartitionedCall"^dense_329/StatefulPartitionedCall,^prediction_output_0/StatefulPartitionedCall.^spatial_dropout2d_109/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:         Й:         ЙЙ: : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_109/StatefulPartitionedCall/batch_normalization_109/StatefulPartitionedCall2H
"conv2d_654/StatefulPartitionedCall"conv2d_654/StatefulPartitionedCall2H
"conv2d_655/StatefulPartitionedCall"conv2d_655/StatefulPartitionedCall2H
"conv2d_656/StatefulPartitionedCall"conv2d_656/StatefulPartitionedCall2H
"conv2d_657/StatefulPartitionedCall"conv2d_657/StatefulPartitionedCall2H
"conv2d_658/StatefulPartitionedCall"conv2d_658/StatefulPartitionedCall2H
"conv2d_659/StatefulPartitionedCall"conv2d_659/StatefulPartitionedCall2F
!dense_327/StatefulPartitionedCall!dense_327/StatefulPartitionedCall2F
!dense_328/StatefulPartitionedCall!dense_328/StatefulPartitionedCall2F
!dense_329/StatefulPartitionedCall!dense_329/StatefulPartitionedCall2Z
+prediction_output_0/StatefulPartitionedCall+prediction_output_0/StatefulPartitionedCall2^
-spatial_dropout2d_109/StatefulPartitionedCall-spatial_dropout2d_109/StatefulPartitionedCall:[ W
0
_output_shapes
:         Й
#
_user_specified_name	input_219:\X
1
_output_shapes
:         ЙЙ
#
_user_specified_name	input_220
█
┤
-__inference_joint_model_layer_call_fn_4210354
	input_219
	input_220!
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
StatefulPartitionedCallStatefulPartitionedCall	input_219	input_220unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
H__inference_joint_model_layer_call_and_return_conditional_losses_4210249o
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
_user_specified_name	input_219:\X
1
_output_shapes
:         ЙЙ
#
_user_specified_name	input_220
Ъ	
╘
9__inference_batch_normalization_109_layer_call_fn_4210985

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallа
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
GPU2*0J 8В *]
fXRV
T__inference_batch_normalization_109_layer_call_and_return_conditional_losses_4209683Й
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
М
А
G__inference_conv2d_654_layer_call_and_return_conditional_losses_4210952

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
Э

ў
F__inference_dense_328_layer_call_and_return_conditional_losses_4211180

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
Н
А
G__inference_conv2d_659_layer_call_and_return_conditional_losses_4211054

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
тТ
┤
"__inference__wrapped_model_4209621
	input_219
	input_220O
5joint_model_conv2d_657_conv2d_readvariableop_resource:		D
6joint_model_conv2d_657_biasadd_readvariableop_resource:O
5joint_model_conv2d_654_conv2d_readvariableop_resource:D
6joint_model_conv2d_654_biasadd_readvariableop_resource:I
;joint_model_batch_normalization_109_readvariableop_resource:K
=joint_model_batch_normalization_109_readvariableop_1_resource:Z
Ljoint_model_batch_normalization_109_fusedbatchnormv3_readvariableop_resource:\
Njoint_model_batch_normalization_109_fusedbatchnormv3_readvariableop_1_resource:O
5joint_model_conv2d_658_conv2d_readvariableop_resource:D
6joint_model_conv2d_658_biasadd_readvariableop_resource:O
5joint_model_conv2d_655_conv2d_readvariableop_resource:	D
6joint_model_conv2d_655_biasadd_readvariableop_resource:O
5joint_model_conv2d_659_conv2d_readvariableop_resource:D
6joint_model_conv2d_659_biasadd_readvariableop_resource:O
5joint_model_conv2d_656_conv2d_readvariableop_resource:D
6joint_model_conv2d_656_biasadd_readvariableop_resource:G
4joint_model_dense_327_matmul_readvariableop_resource:	ФC
5joint_model_dense_327_biasadd_readvariableop_resource:F
4joint_model_dense_328_matmul_readvariableop_resource:C
5joint_model_dense_328_biasadd_readvariableop_resource:F
4joint_model_dense_329_matmul_readvariableop_resource:C
5joint_model_dense_329_biasadd_readvariableop_resource:P
>joint_model_prediction_output_0_matmul_readvariableop_resource:M
?joint_model_prediction_output_0_biasadd_readvariableop_resource:
identityИвCjoint_model/batch_normalization_109/FusedBatchNormV3/ReadVariableOpвEjoint_model/batch_normalization_109/FusedBatchNormV3/ReadVariableOp_1в2joint_model/batch_normalization_109/ReadVariableOpв4joint_model/batch_normalization_109/ReadVariableOp_1в-joint_model/conv2d_654/BiasAdd/ReadVariableOpв,joint_model/conv2d_654/Conv2D/ReadVariableOpв-joint_model/conv2d_655/BiasAdd/ReadVariableOpв,joint_model/conv2d_655/Conv2D/ReadVariableOpв-joint_model/conv2d_656/BiasAdd/ReadVariableOpв,joint_model/conv2d_656/Conv2D/ReadVariableOpв-joint_model/conv2d_657/BiasAdd/ReadVariableOpв,joint_model/conv2d_657/Conv2D/ReadVariableOpв-joint_model/conv2d_658/BiasAdd/ReadVariableOpв,joint_model/conv2d_658/Conv2D/ReadVariableOpв-joint_model/conv2d_659/BiasAdd/ReadVariableOpв,joint_model/conv2d_659/Conv2D/ReadVariableOpв,joint_model/dense_327/BiasAdd/ReadVariableOpв+joint_model/dense_327/MatMul/ReadVariableOpв,joint_model/dense_328/BiasAdd/ReadVariableOpв+joint_model/dense_328/MatMul/ReadVariableOpв,joint_model/dense_329/BiasAdd/ReadVariableOpв+joint_model/dense_329/MatMul/ReadVariableOpв6joint_model/prediction_output_0/BiasAdd/ReadVariableOpв5joint_model/prediction_output_0/MatMul/ReadVariableOpк
,joint_model/conv2d_657/Conv2D/ReadVariableOpReadVariableOp5joint_model_conv2d_657_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0╠
joint_model/conv2d_657/Conv2DConv2D	input_2204joint_model/conv2d_657/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides	
Йа
-joint_model/conv2d_657/BiasAdd/ReadVariableOpReadVariableOp6joint_model_conv2d_657_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0├
joint_model/conv2d_657/BiasAddBiasAdd&joint_model/conv2d_657/Conv2D:output:05joint_model/conv2d_657/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ЙЗ
joint_model/conv2d_657/ReluRelu'joint_model/conv2d_657/BiasAdd:output:0*
T0*0
_output_shapes
:         Йк
,joint_model/conv2d_654/Conv2D/ReadVariableOpReadVariableOp5joint_model_conv2d_654_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╦
joint_model/conv2d_654/Conv2DConv2D	input_2194joint_model/conv2d_654/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides
а
-joint_model/conv2d_654/BiasAdd/ReadVariableOpReadVariableOp6joint_model_conv2d_654_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0├
joint_model/conv2d_654/BiasAddBiasAdd&joint_model/conv2d_654/Conv2D:output:05joint_model/conv2d_654/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ЙЗ
joint_model/conv2d_654/ReluRelu'joint_model/conv2d_654/BiasAdd:output:0*
T0*0
_output_shapes
:         ЙЬ
*joint_model/spatial_dropout2d_109/IdentityIdentity)joint_model/conv2d_657/Relu:activations:0*
T0*0
_output_shapes
:         Йк
2joint_model/batch_normalization_109/ReadVariableOpReadVariableOp;joint_model_batch_normalization_109_readvariableop_resource*
_output_shapes
:*
dtype0о
4joint_model/batch_normalization_109/ReadVariableOp_1ReadVariableOp=joint_model_batch_normalization_109_readvariableop_1_resource*
_output_shapes
:*
dtype0╠
Cjoint_model/batch_normalization_109/FusedBatchNormV3/ReadVariableOpReadVariableOpLjoint_model_batch_normalization_109_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╨
Ejoint_model/batch_normalization_109/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNjoint_model_batch_normalization_109_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0О
4joint_model/batch_normalization_109/FusedBatchNormV3FusedBatchNormV3)joint_model/conv2d_654/Relu:activations:0:joint_model/batch_normalization_109/ReadVariableOp:value:0<joint_model/batch_normalization_109/ReadVariableOp_1:value:0Kjoint_model/batch_normalization_109/FusedBatchNormV3/ReadVariableOp:value:0Mjoint_model/batch_normalization_109/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         Й:::::*
epsilon%oГ:*
is_training( к
,joint_model/conv2d_658/Conv2D/ReadVariableOpReadVariableOp5joint_model_conv2d_658_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ў
joint_model/conv2d_658/Conv2DConv2D3joint_model/spatial_dropout2d_109/Identity:output:04joint_model/conv2d_658/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides	
Йа
-joint_model/conv2d_658/BiasAdd/ReadVariableOpReadVariableOp6joint_model_conv2d_658_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0├
joint_model/conv2d_658/BiasAddBiasAdd&joint_model/conv2d_658/Conv2D:output:05joint_model/conv2d_658/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ЙЗ
joint_model/conv2d_658/ReluRelu'joint_model/conv2d_658/BiasAdd:output:0*
T0*0
_output_shapes
:         Йк
,joint_model/conv2d_655/Conv2D/ReadVariableOpReadVariableOp5joint_model_conv2d_655_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0·
joint_model/conv2d_655/Conv2DConv2D8joint_model/batch_normalization_109/FusedBatchNormV3:y:04joint_model/conv2d_655/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides
а
-joint_model/conv2d_655/BiasAdd/ReadVariableOpReadVariableOp6joint_model_conv2d_655_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0├
joint_model/conv2d_655/BiasAddBiasAdd&joint_model/conv2d_655/Conv2D:output:05joint_model/conv2d_655/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ЙЗ
joint_model/conv2d_655/ReluRelu'joint_model/conv2d_655/BiasAdd:output:0*
T0*0
_output_shapes
:         Йк
,joint_model/conv2d_659/Conv2D/ReadVariableOpReadVariableOp5joint_model_conv2d_659_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ь
joint_model/conv2d_659/Conv2DConv2D)joint_model/conv2d_658/Relu:activations:04joint_model/conv2d_659/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides	
Йа
-joint_model/conv2d_659/BiasAdd/ReadVariableOpReadVariableOp6joint_model_conv2d_659_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0├
joint_model/conv2d_659/BiasAddBiasAdd&joint_model/conv2d_659/Conv2D:output:05joint_model/conv2d_659/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ЙЗ
joint_model/conv2d_659/ReluRelu'joint_model/conv2d_659/BiasAdd:output:0*
T0*0
_output_shapes
:         Йк
,joint_model/conv2d_656/Conv2D/ReadVariableOpReadVariableOp5joint_model_conv2d_656_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ы
joint_model/conv2d_656/Conv2DConv2D)joint_model/conv2d_655/Relu:activations:04joint_model/conv2d_656/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides
а
-joint_model/conv2d_656/BiasAdd/ReadVariableOpReadVariableOp6joint_model_conv2d_656_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0├
joint_model/conv2d_656/BiasAddBiasAdd&joint_model/conv2d_656/Conv2D:output:05joint_model/conv2d_656/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ЙЗ
joint_model/conv2d_656/ReluRelu'joint_model/conv2d_656/BiasAdd:output:0*
T0*0
_output_shapes
:         ЙЛ
:joint_model/global_max_pooling2d_109/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ╤
(joint_model/global_max_pooling2d_109/MaxMax)joint_model/conv2d_659/Relu:activations:0Cjoint_model/global_max_pooling2d_109/Max/reduction_indices:output:0*
T0*'
_output_shapes
:         n
joint_model/flatten_219/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ╖
joint_model/flatten_219/ReshapeReshape1joint_model/global_max_pooling2d_109/Max:output:0&joint_model/flatten_219/Const:output:0*
T0*'
_output_shapes
:         n
joint_model/flatten_218/ConstConst*
_output_shapes
:*
dtype0*
valueB"      ░
joint_model/flatten_218/ReshapeReshape)joint_model/conv2d_656/Relu:activations:0&joint_model/flatten_218/Const:output:0*
T0*(
_output_shapes
:         Тi
'joint_model/concatenate_109/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ё
"joint_model/concatenate_109/concatConcatV2(joint_model/flatten_219/Reshape:output:0(joint_model/flatten_218/Reshape:output:00joint_model/concatenate_109/concat/axis:output:0*
N*
T0*(
_output_shapes
:         Фб
+joint_model/dense_327/MatMul/ReadVariableOpReadVariableOp4joint_model_dense_327_matmul_readvariableop_resource*
_output_shapes
:	Ф*
dtype0║
joint_model/dense_327/MatMulMatMul+joint_model/concatenate_109/concat:output:03joint_model/dense_327/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ю
,joint_model/dense_327/BiasAdd/ReadVariableOpReadVariableOp5joint_model_dense_327_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╕
joint_model/dense_327/BiasAddBiasAdd&joint_model/dense_327/MatMul:product:04joint_model/dense_327/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         |
joint_model/dense_327/ReluRelu&joint_model/dense_327/BiasAdd:output:0*
T0*'
_output_shapes
:         а
+joint_model/dense_328/MatMul/ReadVariableOpReadVariableOp4joint_model_dense_328_matmul_readvariableop_resource*
_output_shapes

:*
dtype0╖
joint_model/dense_328/MatMulMatMul(joint_model/dense_327/Relu:activations:03joint_model/dense_328/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ю
,joint_model/dense_328/BiasAdd/ReadVariableOpReadVariableOp5joint_model_dense_328_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╕
joint_model/dense_328/BiasAddBiasAdd&joint_model/dense_328/MatMul:product:04joint_model/dense_328/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         |
joint_model/dense_328/ReluRelu&joint_model/dense_328/BiasAdd:output:0*
T0*'
_output_shapes
:         а
+joint_model/dense_329/MatMul/ReadVariableOpReadVariableOp4joint_model_dense_329_matmul_readvariableop_resource*
_output_shapes

:*
dtype0╖
joint_model/dense_329/MatMulMatMul(joint_model/dense_328/Relu:activations:03joint_model/dense_329/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ю
,joint_model/dense_329/BiasAdd/ReadVariableOpReadVariableOp5joint_model_dense_329_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╕
joint_model/dense_329/BiasAddBiasAdd&joint_model/dense_329/MatMul:product:04joint_model/dense_329/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         |
joint_model/dense_329/ReluRelu&joint_model/dense_329/BiasAdd:output:0*
T0*'
_output_shapes
:         ┤
5joint_model/prediction_output_0/MatMul/ReadVariableOpReadVariableOp>joint_model_prediction_output_0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0╦
&joint_model/prediction_output_0/MatMulMatMul(joint_model/dense_329/Relu:activations:0=joint_model/prediction_output_0/MatMul/ReadVariableOp:value:0*
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
:         В

NoOpNoOpD^joint_model/batch_normalization_109/FusedBatchNormV3/ReadVariableOpF^joint_model/batch_normalization_109/FusedBatchNormV3/ReadVariableOp_13^joint_model/batch_normalization_109/ReadVariableOp5^joint_model/batch_normalization_109/ReadVariableOp_1.^joint_model/conv2d_654/BiasAdd/ReadVariableOp-^joint_model/conv2d_654/Conv2D/ReadVariableOp.^joint_model/conv2d_655/BiasAdd/ReadVariableOp-^joint_model/conv2d_655/Conv2D/ReadVariableOp.^joint_model/conv2d_656/BiasAdd/ReadVariableOp-^joint_model/conv2d_656/Conv2D/ReadVariableOp.^joint_model/conv2d_657/BiasAdd/ReadVariableOp-^joint_model/conv2d_657/Conv2D/ReadVariableOp.^joint_model/conv2d_658/BiasAdd/ReadVariableOp-^joint_model/conv2d_658/Conv2D/ReadVariableOp.^joint_model/conv2d_659/BiasAdd/ReadVariableOp-^joint_model/conv2d_659/Conv2D/ReadVariableOp-^joint_model/dense_327/BiasAdd/ReadVariableOp,^joint_model/dense_327/MatMul/ReadVariableOp-^joint_model/dense_328/BiasAdd/ReadVariableOp,^joint_model/dense_328/MatMul/ReadVariableOp-^joint_model/dense_329/BiasAdd/ReadVariableOp,^joint_model/dense_329/MatMul/ReadVariableOp7^joint_model/prediction_output_0/BiasAdd/ReadVariableOp6^joint_model/prediction_output_0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:         Й:         ЙЙ: : : : : : : : : : : : : : : : : : : : : : : : 2К
Cjoint_model/batch_normalization_109/FusedBatchNormV3/ReadVariableOpCjoint_model/batch_normalization_109/FusedBatchNormV3/ReadVariableOp2О
Ejoint_model/batch_normalization_109/FusedBatchNormV3/ReadVariableOp_1Ejoint_model/batch_normalization_109/FusedBatchNormV3/ReadVariableOp_12h
2joint_model/batch_normalization_109/ReadVariableOp2joint_model/batch_normalization_109/ReadVariableOp2l
4joint_model/batch_normalization_109/ReadVariableOp_14joint_model/batch_normalization_109/ReadVariableOp_12^
-joint_model/conv2d_654/BiasAdd/ReadVariableOp-joint_model/conv2d_654/BiasAdd/ReadVariableOp2\
,joint_model/conv2d_654/Conv2D/ReadVariableOp,joint_model/conv2d_654/Conv2D/ReadVariableOp2^
-joint_model/conv2d_655/BiasAdd/ReadVariableOp-joint_model/conv2d_655/BiasAdd/ReadVariableOp2\
,joint_model/conv2d_655/Conv2D/ReadVariableOp,joint_model/conv2d_655/Conv2D/ReadVariableOp2^
-joint_model/conv2d_656/BiasAdd/ReadVariableOp-joint_model/conv2d_656/BiasAdd/ReadVariableOp2\
,joint_model/conv2d_656/Conv2D/ReadVariableOp,joint_model/conv2d_656/Conv2D/ReadVariableOp2^
-joint_model/conv2d_657/BiasAdd/ReadVariableOp-joint_model/conv2d_657/BiasAdd/ReadVariableOp2\
,joint_model/conv2d_657/Conv2D/ReadVariableOp,joint_model/conv2d_657/Conv2D/ReadVariableOp2^
-joint_model/conv2d_658/BiasAdd/ReadVariableOp-joint_model/conv2d_658/BiasAdd/ReadVariableOp2\
,joint_model/conv2d_658/Conv2D/ReadVariableOp,joint_model/conv2d_658/Conv2D/ReadVariableOp2^
-joint_model/conv2d_659/BiasAdd/ReadVariableOp-joint_model/conv2d_659/BiasAdd/ReadVariableOp2\
,joint_model/conv2d_659/Conv2D/ReadVariableOp,joint_model/conv2d_659/Conv2D/ReadVariableOp2\
,joint_model/dense_327/BiasAdd/ReadVariableOp,joint_model/dense_327/BiasAdd/ReadVariableOp2Z
+joint_model/dense_327/MatMul/ReadVariableOp+joint_model/dense_327/MatMul/ReadVariableOp2\
,joint_model/dense_328/BiasAdd/ReadVariableOp,joint_model/dense_328/BiasAdd/ReadVariableOp2Z
+joint_model/dense_328/MatMul/ReadVariableOp+joint_model/dense_328/MatMul/ReadVariableOp2\
,joint_model/dense_329/BiasAdd/ReadVariableOp,joint_model/dense_329/BiasAdd/ReadVariableOp2Z
+joint_model/dense_329/MatMul/ReadVariableOp+joint_model/dense_329/MatMul/ReadVariableOp2p
6joint_model/prediction_output_0/BiasAdd/ReadVariableOp6joint_model/prediction_output_0/BiasAdd/ReadVariableOp2n
5joint_model/prediction_output_0/MatMul/ReadVariableOp5joint_model/prediction_output_0/MatMul/ReadVariableOp:[ W
0
_output_shapes
:         Й
#
_user_specified_name	input_219:\X
1
_output_shapes
:         ЙЙ
#
_user_specified_name	input_220
ў
б
,__inference_conv2d_654_layer_call_fn_4210941

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
G__inference_conv2d_654_layer_call_and_return_conditional_losses_4209775x
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
М
А
G__inference_conv2d_655_layer_call_and_return_conditional_losses_4209819

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
╔
x
L__inference_concatenate_109_layer_call_and_return_conditional_losses_4211140
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
▒
q
U__inference_global_max_pooling2d_109_layer_call_and_return_conditional_losses_4211085

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
╧
Я
T__inference_batch_normalization_109_layer_call_and_return_conditional_losses_4209683

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
ў
б
,__inference_conv2d_658_layer_call_fn_4210961

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
G__inference_conv2d_658_layer_call_and_return_conditional_losses_4209802x
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
▒
q
U__inference_global_max_pooling2d_109_layer_call_and_return_conditional_losses_4209735

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
╠
d
H__inference_flatten_218_layer_call_and_return_conditional_losses_4211127

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
М
А
G__inference_conv2d_655_layer_call_and_return_conditional_losses_4211074

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
╕P
Ш
H__inference_joint_model_layer_call_and_return_conditional_losses_4210249

inputs
inputs_1,
conv2d_657_4210184:		 
conv2d_657_4210186:,
conv2d_654_4210189: 
conv2d_654_4210191:-
batch_normalization_109_4210195:-
batch_normalization_109_4210197:-
batch_normalization_109_4210199:-
batch_normalization_109_4210201:,
conv2d_658_4210204: 
conv2d_658_4210206:,
conv2d_655_4210209:	 
conv2d_655_4210211:,
conv2d_659_4210214: 
conv2d_659_4210216:,
conv2d_656_4210219: 
conv2d_656_4210221:$
dense_327_4210228:	Ф
dense_327_4210230:#
dense_328_4210233:
dense_328_4210235:#
dense_329_4210238:
dense_329_4210240:-
prediction_output_0_4210243:)
prediction_output_0_4210245:
identityИв/batch_normalization_109/StatefulPartitionedCallв"conv2d_654/StatefulPartitionedCallв"conv2d_655/StatefulPartitionedCallв"conv2d_656/StatefulPartitionedCallв"conv2d_657/StatefulPartitionedCallв"conv2d_658/StatefulPartitionedCallв"conv2d_659/StatefulPartitionedCallв!dense_327/StatefulPartitionedCallв!dense_328/StatefulPartitionedCallв!dense_329/StatefulPartitionedCallв+prediction_output_0/StatefulPartitionedCallв-spatial_dropout2d_109/StatefulPartitionedCallЙ
"conv2d_657/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv2d_657_4210184conv2d_657_4210186*
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
G__inference_conv2d_657_layer_call_and_return_conditional_losses_4209758З
"conv2d_654/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_654_4210189conv2d_654_4210191*
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
G__inference_conv2d_654_layer_call_and_return_conditional_losses_4209775Ф
-spatial_dropout2d_109/StatefulPartitionedCallStatefulPartitionedCall+conv2d_657/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *[
fVRT
R__inference_spatial_dropout2d_109_layer_call_and_return_conditional_losses_4209658д
/batch_normalization_109/StatefulPartitionedCallStatefulPartitionedCall+conv2d_654/StatefulPartitionedCall:output:0batch_normalization_109_4210195batch_normalization_109_4210197batch_normalization_109_4210199batch_normalization_109_4210201*
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
GPU2*0J 8В *]
fXRV
T__inference_batch_normalization_109_layer_call_and_return_conditional_losses_4209714╖
"conv2d_658/StatefulPartitionedCallStatefulPartitionedCall6spatial_dropout2d_109/StatefulPartitionedCall:output:0conv2d_658_4210204conv2d_658_4210206*
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
G__inference_conv2d_658_layer_call_and_return_conditional_losses_4209802╣
"conv2d_655/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_109/StatefulPartitionedCall:output:0conv2d_655_4210209conv2d_655_4210211*
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
G__inference_conv2d_655_layer_call_and_return_conditional_losses_4209819м
"conv2d_659/StatefulPartitionedCallStatefulPartitionedCall+conv2d_658/StatefulPartitionedCall:output:0conv2d_659_4210214conv2d_659_4210216*
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
G__inference_conv2d_659_layer_call_and_return_conditional_losses_4209836м
"conv2d_656/StatefulPartitionedCallStatefulPartitionedCall+conv2d_655/StatefulPartitionedCall:output:0conv2d_656_4210219conv2d_656_4210221*
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
G__inference_conv2d_656_layer_call_and_return_conditional_losses_4209853Б
(global_max_pooling2d_109/PartitionedCallPartitionedCall+conv2d_659/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *^
fYRW
U__inference_global_max_pooling2d_109_layer_call_and_return_conditional_losses_4209735э
flatten_219/PartitionedCallPartitionedCall1global_max_pooling2d_109/PartitionedCall:output:0*
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
H__inference_flatten_219_layer_call_and_return_conditional_losses_4209866ш
flatten_218/PartitionedCallPartitionedCall+conv2d_656/StatefulPartitionedCall:output:0*
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
H__inference_flatten_218_layer_call_and_return_conditional_losses_4209874Р
concatenate_109/PartitionedCallPartitionedCall$flatten_219/PartitionedCall:output:0$flatten_218/PartitionedCall:output:0*
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
GPU2*0J 8В *U
fPRN
L__inference_concatenate_109_layer_call_and_return_conditional_losses_4209883Ь
!dense_327/StatefulPartitionedCallStatefulPartitionedCall(concatenate_109/PartitionedCall:output:0dense_327_4210228dense_327_4210230*
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
F__inference_dense_327_layer_call_and_return_conditional_losses_4209896Ю
!dense_328/StatefulPartitionedCallStatefulPartitionedCall*dense_327/StatefulPartitionedCall:output:0dense_328_4210233dense_328_4210235*
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
F__inference_dense_328_layer_call_and_return_conditional_losses_4209913Ю
!dense_329/StatefulPartitionedCallStatefulPartitionedCall*dense_328/StatefulPartitionedCall:output:0dense_329_4210238dense_329_4210240*
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
F__inference_dense_329_layer_call_and_return_conditional_losses_4209930╞
+prediction_output_0/StatefulPartitionedCallStatefulPartitionedCall*dense_329/StatefulPartitionedCall:output:0prediction_output_0_4210243prediction_output_0_4210245*
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
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4209946Г
IdentityIdentity4prediction_output_0/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         а
NoOpNoOp0^batch_normalization_109/StatefulPartitionedCall#^conv2d_654/StatefulPartitionedCall#^conv2d_655/StatefulPartitionedCall#^conv2d_656/StatefulPartitionedCall#^conv2d_657/StatefulPartitionedCall#^conv2d_658/StatefulPartitionedCall#^conv2d_659/StatefulPartitionedCall"^dense_327/StatefulPartitionedCall"^dense_328/StatefulPartitionedCall"^dense_329/StatefulPartitionedCall,^prediction_output_0/StatefulPartitionedCall.^spatial_dropout2d_109/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:         Й:         ЙЙ: : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_109/StatefulPartitionedCall/batch_normalization_109/StatefulPartitionedCall2H
"conv2d_654/StatefulPartitionedCall"conv2d_654/StatefulPartitionedCall2H
"conv2d_655/StatefulPartitionedCall"conv2d_655/StatefulPartitionedCall2H
"conv2d_656/StatefulPartitionedCall"conv2d_656/StatefulPartitionedCall2H
"conv2d_657/StatefulPartitionedCall"conv2d_657/StatefulPartitionedCall2H
"conv2d_658/StatefulPartitionedCall"conv2d_658/StatefulPartitionedCall2H
"conv2d_659/StatefulPartitionedCall"conv2d_659/StatefulPartitionedCall2F
!dense_327/StatefulPartitionedCall!dense_327/StatefulPartitionedCall2F
!dense_328/StatefulPartitionedCall!dense_328/StatefulPartitionedCall2F
!dense_329/StatefulPartitionedCall!dense_329/StatefulPartitionedCall2Z
+prediction_output_0/StatefulPartitionedCall+prediction_output_0/StatefulPartitionedCall2^
-spatial_dropout2d_109/StatefulPartitionedCall-spatial_dropout2d_109/StatefulPartitionedCall:X T
0
_output_shapes
:         Й
 
_user_specified_nameinputs:YU
1
_output_shapes
:         ЙЙ
 
_user_specified_nameinputs
╠
d
H__inference_flatten_218_layer_call_and_return_conditional_losses_4209874

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
╤y
Ш
H__inference_joint_model_layer_call_and_return_conditional_losses_4210759
inputs_0
inputs_1C
)conv2d_657_conv2d_readvariableop_resource:		8
*conv2d_657_biasadd_readvariableop_resource:C
)conv2d_654_conv2d_readvariableop_resource:8
*conv2d_654_biasadd_readvariableop_resource:=
/batch_normalization_109_readvariableop_resource:?
1batch_normalization_109_readvariableop_1_resource:N
@batch_normalization_109_fusedbatchnormv3_readvariableop_resource:P
Bbatch_normalization_109_fusedbatchnormv3_readvariableop_1_resource:C
)conv2d_658_conv2d_readvariableop_resource:8
*conv2d_658_biasadd_readvariableop_resource:C
)conv2d_655_conv2d_readvariableop_resource:	8
*conv2d_655_biasadd_readvariableop_resource:C
)conv2d_659_conv2d_readvariableop_resource:8
*conv2d_659_biasadd_readvariableop_resource:C
)conv2d_656_conv2d_readvariableop_resource:8
*conv2d_656_biasadd_readvariableop_resource:;
(dense_327_matmul_readvariableop_resource:	Ф7
)dense_327_biasadd_readvariableop_resource::
(dense_328_matmul_readvariableop_resource:7
)dense_328_biasadd_readvariableop_resource::
(dense_329_matmul_readvariableop_resource:7
)dense_329_biasadd_readvariableop_resource:D
2prediction_output_0_matmul_readvariableop_resource:A
3prediction_output_0_biasadd_readvariableop_resource:
identityИв7batch_normalization_109/FusedBatchNormV3/ReadVariableOpв9batch_normalization_109/FusedBatchNormV3/ReadVariableOp_1в&batch_normalization_109/ReadVariableOpв(batch_normalization_109/ReadVariableOp_1в!conv2d_654/BiasAdd/ReadVariableOpв conv2d_654/Conv2D/ReadVariableOpв!conv2d_655/BiasAdd/ReadVariableOpв conv2d_655/Conv2D/ReadVariableOpв!conv2d_656/BiasAdd/ReadVariableOpв conv2d_656/Conv2D/ReadVariableOpв!conv2d_657/BiasAdd/ReadVariableOpв conv2d_657/Conv2D/ReadVariableOpв!conv2d_658/BiasAdd/ReadVariableOpв conv2d_658/Conv2D/ReadVariableOpв!conv2d_659/BiasAdd/ReadVariableOpв conv2d_659/Conv2D/ReadVariableOpв dense_327/BiasAdd/ReadVariableOpвdense_327/MatMul/ReadVariableOpв dense_328/BiasAdd/ReadVariableOpвdense_328/MatMul/ReadVariableOpв dense_329/BiasAdd/ReadVariableOpвdense_329/MatMul/ReadVariableOpв*prediction_output_0/BiasAdd/ReadVariableOpв)prediction_output_0/MatMul/ReadVariableOpТ
 conv2d_657/Conv2D/ReadVariableOpReadVariableOp)conv2d_657_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0│
conv2d_657/Conv2DConv2Dinputs_1(conv2d_657/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides	
ЙИ
!conv2d_657/BiasAdd/ReadVariableOpReadVariableOp*conv2d_657_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv2d_657/BiasAddBiasAddconv2d_657/Conv2D:output:0)conv2d_657/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Йo
conv2d_657/ReluReluconv2d_657/BiasAdd:output:0*
T0*0
_output_shapes
:         ЙТ
 conv2d_654/Conv2D/ReadVariableOpReadVariableOp)conv2d_654_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0▓
conv2d_654/Conv2DConv2Dinputs_0(conv2d_654/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides
И
!conv2d_654/BiasAdd/ReadVariableOpReadVariableOp*conv2d_654_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv2d_654/BiasAddBiasAddconv2d_654/Conv2D:output:0)conv2d_654/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Йo
conv2d_654/ReluReluconv2d_654/BiasAdd:output:0*
T0*0
_output_shapes
:         ЙД
spatial_dropout2d_109/IdentityIdentityconv2d_657/Relu:activations:0*
T0*0
_output_shapes
:         ЙТ
&batch_normalization_109/ReadVariableOpReadVariableOp/batch_normalization_109_readvariableop_resource*
_output_shapes
:*
dtype0Ц
(batch_normalization_109/ReadVariableOp_1ReadVariableOp1batch_normalization_109_readvariableop_1_resource*
_output_shapes
:*
dtype0┤
7batch_normalization_109/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_109_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╕
9batch_normalization_109/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_109_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╞
(batch_normalization_109/FusedBatchNormV3FusedBatchNormV3conv2d_654/Relu:activations:0.batch_normalization_109/ReadVariableOp:value:00batch_normalization_109/ReadVariableOp_1:value:0?batch_normalization_109/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_109/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         Й:::::*
epsilon%oГ:*
is_training( Т
 conv2d_658/Conv2D/ReadVariableOpReadVariableOp)conv2d_658_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╥
conv2d_658/Conv2DConv2D'spatial_dropout2d_109/Identity:output:0(conv2d_658/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides	
ЙИ
!conv2d_658/BiasAdd/ReadVariableOpReadVariableOp*conv2d_658_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv2d_658/BiasAddBiasAddconv2d_658/Conv2D:output:0)conv2d_658/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Йo
conv2d_658/ReluReluconv2d_658/BiasAdd:output:0*
T0*0
_output_shapes
:         ЙТ
 conv2d_655/Conv2D/ReadVariableOpReadVariableOp)conv2d_655_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0╓
conv2d_655/Conv2DConv2D,batch_normalization_109/FusedBatchNormV3:y:0(conv2d_655/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides
И
!conv2d_655/BiasAdd/ReadVariableOpReadVariableOp*conv2d_655_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv2d_655/BiasAddBiasAddconv2d_655/Conv2D:output:0)conv2d_655/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Йo
conv2d_655/ReluReluconv2d_655/BiasAdd:output:0*
T0*0
_output_shapes
:         ЙТ
 conv2d_659/Conv2D/ReadVariableOpReadVariableOp)conv2d_659_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╚
conv2d_659/Conv2DConv2Dconv2d_658/Relu:activations:0(conv2d_659/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides	
ЙИ
!conv2d_659/BiasAdd/ReadVariableOpReadVariableOp*conv2d_659_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv2d_659/BiasAddBiasAddconv2d_659/Conv2D:output:0)conv2d_659/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Йo
conv2d_659/ReluReluconv2d_659/BiasAdd:output:0*
T0*0
_output_shapes
:         ЙТ
 conv2d_656/Conv2D/ReadVariableOpReadVariableOp)conv2d_656_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╟
conv2d_656/Conv2DConv2Dconv2d_655/Relu:activations:0(conv2d_656/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides
И
!conv2d_656/BiasAdd/ReadVariableOpReadVariableOp*conv2d_656_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv2d_656/BiasAddBiasAddconv2d_656/Conv2D:output:0)conv2d_656/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Йo
conv2d_656/ReluReluconv2d_656/BiasAdd:output:0*
T0*0
_output_shapes
:         Й
.global_max_pooling2d_109/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      н
global_max_pooling2d_109/MaxMaxconv2d_659/Relu:activations:07global_max_pooling2d_109/Max/reduction_indices:output:0*
T0*'
_output_shapes
:         b
flatten_219/ConstConst*
_output_shapes
:*
dtype0*
valueB"       У
flatten_219/ReshapeReshape%global_max_pooling2d_109/Max:output:0flatten_219/Const:output:0*
T0*'
_output_shapes
:         b
flatten_218/ConstConst*
_output_shapes
:*
dtype0*
valueB"      М
flatten_218/ReshapeReshapeconv2d_656/Relu:activations:0flatten_218/Const:output:0*
T0*(
_output_shapes
:         Т]
concatenate_109/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :└
concatenate_109/concatConcatV2flatten_219/Reshape:output:0flatten_218/Reshape:output:0$concatenate_109/concat/axis:output:0*
N*
T0*(
_output_shapes
:         ФЙ
dense_327/MatMul/ReadVariableOpReadVariableOp(dense_327_matmul_readvariableop_resource*
_output_shapes
:	Ф*
dtype0Ц
dense_327/MatMulMatMulconcatenate_109/concat:output:0'dense_327/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ж
 dense_327/BiasAdd/ReadVariableOpReadVariableOp)dense_327_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
dense_327/BiasAddBiasAdddense_327/MatMul:product:0(dense_327/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d
dense_327/ReluReludense_327/BiasAdd:output:0*
T0*'
_output_shapes
:         И
dense_328/MatMul/ReadVariableOpReadVariableOp(dense_328_matmul_readvariableop_resource*
_output_shapes

:*
dtype0У
dense_328/MatMulMatMuldense_327/Relu:activations:0'dense_328/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ж
 dense_328/BiasAdd/ReadVariableOpReadVariableOp)dense_328_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
dense_328/BiasAddBiasAdddense_328/MatMul:product:0(dense_328/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d
dense_328/ReluReludense_328/BiasAdd:output:0*
T0*'
_output_shapes
:         И
dense_329/MatMul/ReadVariableOpReadVariableOp(dense_329_matmul_readvariableop_resource*
_output_shapes

:*
dtype0У
dense_329/MatMulMatMuldense_328/Relu:activations:0'dense_329/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ж
 dense_329/BiasAdd/ReadVariableOpReadVariableOp)dense_329_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
dense_329/BiasAddBiasAdddense_329/MatMul:product:0(dense_329/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d
dense_329/ReluReludense_329/BiasAdd:output:0*
T0*'
_output_shapes
:         Ь
)prediction_output_0/MatMul/ReadVariableOpReadVariableOp2prediction_output_0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0з
prediction_output_0/MatMulMatMuldense_329/Relu:activations:01prediction_output_0/MatMul/ReadVariableOp:value:0*
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
:         т
NoOpNoOp8^batch_normalization_109/FusedBatchNormV3/ReadVariableOp:^batch_normalization_109/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_109/ReadVariableOp)^batch_normalization_109/ReadVariableOp_1"^conv2d_654/BiasAdd/ReadVariableOp!^conv2d_654/Conv2D/ReadVariableOp"^conv2d_655/BiasAdd/ReadVariableOp!^conv2d_655/Conv2D/ReadVariableOp"^conv2d_656/BiasAdd/ReadVariableOp!^conv2d_656/Conv2D/ReadVariableOp"^conv2d_657/BiasAdd/ReadVariableOp!^conv2d_657/Conv2D/ReadVariableOp"^conv2d_658/BiasAdd/ReadVariableOp!^conv2d_658/Conv2D/ReadVariableOp"^conv2d_659/BiasAdd/ReadVariableOp!^conv2d_659/Conv2D/ReadVariableOp!^dense_327/BiasAdd/ReadVariableOp ^dense_327/MatMul/ReadVariableOp!^dense_328/BiasAdd/ReadVariableOp ^dense_328/MatMul/ReadVariableOp!^dense_329/BiasAdd/ReadVariableOp ^dense_329/MatMul/ReadVariableOp+^prediction_output_0/BiasAdd/ReadVariableOp*^prediction_output_0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:         Й:         ЙЙ: : : : : : : : : : : : : : : : : : : : : : : : 2r
7batch_normalization_109/FusedBatchNormV3/ReadVariableOp7batch_normalization_109/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_109/FusedBatchNormV3/ReadVariableOp_19batch_normalization_109/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_109/ReadVariableOp&batch_normalization_109/ReadVariableOp2T
(batch_normalization_109/ReadVariableOp_1(batch_normalization_109/ReadVariableOp_12F
!conv2d_654/BiasAdd/ReadVariableOp!conv2d_654/BiasAdd/ReadVariableOp2D
 conv2d_654/Conv2D/ReadVariableOp conv2d_654/Conv2D/ReadVariableOp2F
!conv2d_655/BiasAdd/ReadVariableOp!conv2d_655/BiasAdd/ReadVariableOp2D
 conv2d_655/Conv2D/ReadVariableOp conv2d_655/Conv2D/ReadVariableOp2F
!conv2d_656/BiasAdd/ReadVariableOp!conv2d_656/BiasAdd/ReadVariableOp2D
 conv2d_656/Conv2D/ReadVariableOp conv2d_656/Conv2D/ReadVariableOp2F
!conv2d_657/BiasAdd/ReadVariableOp!conv2d_657/BiasAdd/ReadVariableOp2D
 conv2d_657/Conv2D/ReadVariableOp conv2d_657/Conv2D/ReadVariableOp2F
!conv2d_658/BiasAdd/ReadVariableOp!conv2d_658/BiasAdd/ReadVariableOp2D
 conv2d_658/Conv2D/ReadVariableOp conv2d_658/Conv2D/ReadVariableOp2F
!conv2d_659/BiasAdd/ReadVariableOp!conv2d_659/BiasAdd/ReadVariableOp2D
 conv2d_659/Conv2D/ReadVariableOp conv2d_659/Conv2D/ReadVariableOp2D
 dense_327/BiasAdd/ReadVariableOp dense_327/BiasAdd/ReadVariableOp2B
dense_327/MatMul/ReadVariableOpdense_327/MatMul/ReadVariableOp2D
 dense_328/BiasAdd/ReadVariableOp dense_328/BiasAdd/ReadVariableOp2B
dense_328/MatMul/ReadVariableOpdense_328/MatMul/ReadVariableOp2D
 dense_329/BiasAdd/ReadVariableOp dense_329/BiasAdd/ReadVariableOp2B
dense_329/MatMul/ReadVariableOpdense_329/MatMul/ReadVariableOp2X
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
╕
d
H__inference_flatten_219_layer_call_and_return_conditional_losses_4209866

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
▌
в
5__inference_prediction_output_0_layer_call_fn_4211209

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
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4209946o
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
Б
q
R__inference_spatial_dropout2d_109_layer_call_and_return_conditional_losses_4209658

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
б

°
F__inference_dense_327_layer_call_and_return_conditional_losses_4209896

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
╠
Щ
+__inference_dense_327_layer_call_fn_4211149

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
F__inference_dense_327_layer_call_and_return_conditional_losses_4209896o
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
Н
А
G__inference_conv2d_659_layer_call_and_return_conditional_losses_4209836

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
-__inference_joint_model_layer_call_fn_4210662
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
H__inference_joint_model_layer_call_and_return_conditional_losses_4210249o
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
П
А
G__inference_conv2d_657_layer_call_and_return_conditional_losses_4209758

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
╔
S
7__inference_spatial_dropout2d_109_layer_call_fn_4210899

inputs
identityу
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
GPU2*0J 8В *[
fVRT
R__inference_spatial_dropout2d_109_layer_call_and_return_conditional_losses_4209630Г
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
М
А
G__inference_conv2d_656_layer_call_and_return_conditional_losses_4211105

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
ЗХ
╓ 
 __inference__traced_save_4211468
file_prefix0
,savev2_conv2d_657_kernel_read_readvariableop.
*savev2_conv2d_657_bias_read_readvariableop0
,savev2_conv2d_654_kernel_read_readvariableop.
*savev2_conv2d_654_bias_read_readvariableop0
,savev2_conv2d_658_kernel_read_readvariableop.
*savev2_conv2d_658_bias_read_readvariableop<
8savev2_batch_normalization_109_gamma_read_readvariableop;
7savev2_batch_normalization_109_beta_read_readvariableopB
>savev2_batch_normalization_109_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_109_moving_variance_read_readvariableop0
,savev2_conv2d_659_kernel_read_readvariableop.
*savev2_conv2d_659_bias_read_readvariableop0
,savev2_conv2d_655_kernel_read_readvariableop.
*savev2_conv2d_655_bias_read_readvariableop0
,savev2_conv2d_656_kernel_read_readvariableop.
*savev2_conv2d_656_bias_read_readvariableop/
+savev2_dense_327_kernel_read_readvariableop-
)savev2_dense_327_bias_read_readvariableop/
+savev2_dense_328_kernel_read_readvariableop-
)savev2_dense_328_bias_read_readvariableop/
+savev2_dense_329_kernel_read_readvariableop-
)savev2_dense_329_bias_read_readvariableop9
5savev2_prediction_output_0_kernel_read_readvariableop7
3savev2_prediction_output_0_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop7
3savev2_adam_conv2d_657_kernel_m_read_readvariableop5
1savev2_adam_conv2d_657_bias_m_read_readvariableop7
3savev2_adam_conv2d_654_kernel_m_read_readvariableop5
1savev2_adam_conv2d_654_bias_m_read_readvariableop7
3savev2_adam_conv2d_658_kernel_m_read_readvariableop5
1savev2_adam_conv2d_658_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_109_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_109_beta_m_read_readvariableop7
3savev2_adam_conv2d_659_kernel_m_read_readvariableop5
1savev2_adam_conv2d_659_bias_m_read_readvariableop7
3savev2_adam_conv2d_655_kernel_m_read_readvariableop5
1savev2_adam_conv2d_655_bias_m_read_readvariableop7
3savev2_adam_conv2d_656_kernel_m_read_readvariableop5
1savev2_adam_conv2d_656_bias_m_read_readvariableop6
2savev2_adam_dense_327_kernel_m_read_readvariableop4
0savev2_adam_dense_327_bias_m_read_readvariableop6
2savev2_adam_dense_328_kernel_m_read_readvariableop4
0savev2_adam_dense_328_bias_m_read_readvariableop6
2savev2_adam_dense_329_kernel_m_read_readvariableop4
0savev2_adam_dense_329_bias_m_read_readvariableop@
<savev2_adam_prediction_output_0_kernel_m_read_readvariableop>
:savev2_adam_prediction_output_0_bias_m_read_readvariableop7
3savev2_adam_conv2d_657_kernel_v_read_readvariableop5
1savev2_adam_conv2d_657_bias_v_read_readvariableop7
3savev2_adam_conv2d_654_kernel_v_read_readvariableop5
1savev2_adam_conv2d_654_bias_v_read_readvariableop7
3savev2_adam_conv2d_658_kernel_v_read_readvariableop5
1savev2_adam_conv2d_658_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_109_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_109_beta_v_read_readvariableop7
3savev2_adam_conv2d_659_kernel_v_read_readvariableop5
1savev2_adam_conv2d_659_bias_v_read_readvariableop7
3savev2_adam_conv2d_655_kernel_v_read_readvariableop5
1savev2_adam_conv2d_655_bias_v_read_readvariableop7
3savev2_adam_conv2d_656_kernel_v_read_readvariableop5
1savev2_adam_conv2d_656_bias_v_read_readvariableop6
2savev2_adam_dense_327_kernel_v_read_readvariableop4
0savev2_adam_dense_327_bias_v_read_readvariableop6
2savev2_adam_dense_328_kernel_v_read_readvariableop4
0savev2_adam_dense_328_bias_v_read_readvariableop6
2savev2_adam_dense_329_kernel_v_read_readvariableop4
0savev2_adam_dense_329_bias_v_read_readvariableop@
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
valueгBаLB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ╕
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv2d_657_kernel_read_readvariableop*savev2_conv2d_657_bias_read_readvariableop,savev2_conv2d_654_kernel_read_readvariableop*savev2_conv2d_654_bias_read_readvariableop,savev2_conv2d_658_kernel_read_readvariableop*savev2_conv2d_658_bias_read_readvariableop8savev2_batch_normalization_109_gamma_read_readvariableop7savev2_batch_normalization_109_beta_read_readvariableop>savev2_batch_normalization_109_moving_mean_read_readvariableopBsavev2_batch_normalization_109_moving_variance_read_readvariableop,savev2_conv2d_659_kernel_read_readvariableop*savev2_conv2d_659_bias_read_readvariableop,savev2_conv2d_655_kernel_read_readvariableop*savev2_conv2d_655_bias_read_readvariableop,savev2_conv2d_656_kernel_read_readvariableop*savev2_conv2d_656_bias_read_readvariableop+savev2_dense_327_kernel_read_readvariableop)savev2_dense_327_bias_read_readvariableop+savev2_dense_328_kernel_read_readvariableop)savev2_dense_328_bias_read_readvariableop+savev2_dense_329_kernel_read_readvariableop)savev2_dense_329_bias_read_readvariableop5savev2_prediction_output_0_kernel_read_readvariableop3savev2_prediction_output_0_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop3savev2_adam_conv2d_657_kernel_m_read_readvariableop1savev2_adam_conv2d_657_bias_m_read_readvariableop3savev2_adam_conv2d_654_kernel_m_read_readvariableop1savev2_adam_conv2d_654_bias_m_read_readvariableop3savev2_adam_conv2d_658_kernel_m_read_readvariableop1savev2_adam_conv2d_658_bias_m_read_readvariableop?savev2_adam_batch_normalization_109_gamma_m_read_readvariableop>savev2_adam_batch_normalization_109_beta_m_read_readvariableop3savev2_adam_conv2d_659_kernel_m_read_readvariableop1savev2_adam_conv2d_659_bias_m_read_readvariableop3savev2_adam_conv2d_655_kernel_m_read_readvariableop1savev2_adam_conv2d_655_bias_m_read_readvariableop3savev2_adam_conv2d_656_kernel_m_read_readvariableop1savev2_adam_conv2d_656_bias_m_read_readvariableop2savev2_adam_dense_327_kernel_m_read_readvariableop0savev2_adam_dense_327_bias_m_read_readvariableop2savev2_adam_dense_328_kernel_m_read_readvariableop0savev2_adam_dense_328_bias_m_read_readvariableop2savev2_adam_dense_329_kernel_m_read_readvariableop0savev2_adam_dense_329_bias_m_read_readvariableop<savev2_adam_prediction_output_0_kernel_m_read_readvariableop:savev2_adam_prediction_output_0_bias_m_read_readvariableop3savev2_adam_conv2d_657_kernel_v_read_readvariableop1savev2_adam_conv2d_657_bias_v_read_readvariableop3savev2_adam_conv2d_654_kernel_v_read_readvariableop1savev2_adam_conv2d_654_bias_v_read_readvariableop3savev2_adam_conv2d_658_kernel_v_read_readvariableop1savev2_adam_conv2d_658_bias_v_read_readvariableop?savev2_adam_batch_normalization_109_gamma_v_read_readvariableop>savev2_adam_batch_normalization_109_beta_v_read_readvariableop3savev2_adam_conv2d_659_kernel_v_read_readvariableop1savev2_adam_conv2d_659_bias_v_read_readvariableop3savev2_adam_conv2d_655_kernel_v_read_readvariableop1savev2_adam_conv2d_655_bias_v_read_readvariableop3savev2_adam_conv2d_656_kernel_v_read_readvariableop1savev2_adam_conv2d_656_bias_v_read_readvariableop2savev2_adam_dense_327_kernel_v_read_readvariableop0savev2_adam_dense_327_bias_v_read_readvariableop2savev2_adam_dense_328_kernel_v_read_readvariableop0savev2_adam_dense_328_bias_v_read_readvariableop2savev2_adam_dense_329_kernel_v_read_readvariableop0savev2_adam_dense_329_bias_v_read_readvariableop<savev2_adam_prediction_output_0_kernel_v_read_readvariableop:savev2_adam_prediction_output_0_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
М
А
G__inference_conv2d_656_layer_call_and_return_conditional_losses_4209853

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
Ъ
V
:__inference_global_max_pooling2d_109_layer_call_fn_4211079

inputs
identity╠
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
GPU2*0J 8В *^
fYRW
U__inference_global_max_pooling2d_109_layer_call_and_return_conditional_losses_4209735i
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
Н
А
G__inference_conv2d_658_layer_call_and_return_conditional_losses_4210972

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
Б
q
R__inference_spatial_dropout2d_109_layer_call_and_return_conditional_losses_4210932

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
тN
ш
H__inference_joint_model_layer_call_and_return_conditional_losses_4209953

inputs
inputs_1,
conv2d_657_4209759:		 
conv2d_657_4209761:,
conv2d_654_4209776: 
conv2d_654_4209778:-
batch_normalization_109_4209782:-
batch_normalization_109_4209784:-
batch_normalization_109_4209786:-
batch_normalization_109_4209788:,
conv2d_658_4209803: 
conv2d_658_4209805:,
conv2d_655_4209820:	 
conv2d_655_4209822:,
conv2d_659_4209837: 
conv2d_659_4209839:,
conv2d_656_4209854: 
conv2d_656_4209856:$
dense_327_4209897:	Ф
dense_327_4209899:#
dense_328_4209914:
dense_328_4209916:#
dense_329_4209931:
dense_329_4209933:-
prediction_output_0_4209947:)
prediction_output_0_4209949:
identityИв/batch_normalization_109/StatefulPartitionedCallв"conv2d_654/StatefulPartitionedCallв"conv2d_655/StatefulPartitionedCallв"conv2d_656/StatefulPartitionedCallв"conv2d_657/StatefulPartitionedCallв"conv2d_658/StatefulPartitionedCallв"conv2d_659/StatefulPartitionedCallв!dense_327/StatefulPartitionedCallв!dense_328/StatefulPartitionedCallв!dense_329/StatefulPartitionedCallв+prediction_output_0/StatefulPartitionedCallЙ
"conv2d_657/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv2d_657_4209759conv2d_657_4209761*
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
G__inference_conv2d_657_layer_call_and_return_conditional_losses_4209758З
"conv2d_654/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_654_4209776conv2d_654_4209778*
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
G__inference_conv2d_654_layer_call_and_return_conditional_losses_4209775Д
%spatial_dropout2d_109/PartitionedCallPartitionedCall+conv2d_657/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *[
fVRT
R__inference_spatial_dropout2d_109_layer_call_and_return_conditional_losses_4209630ж
/batch_normalization_109/StatefulPartitionedCallStatefulPartitionedCall+conv2d_654/StatefulPartitionedCall:output:0batch_normalization_109_4209782batch_normalization_109_4209784batch_normalization_109_4209786batch_normalization_109_4209788*
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
GPU2*0J 8В *]
fXRV
T__inference_batch_normalization_109_layer_call_and_return_conditional_losses_4209683п
"conv2d_658/StatefulPartitionedCallStatefulPartitionedCall.spatial_dropout2d_109/PartitionedCall:output:0conv2d_658_4209803conv2d_658_4209805*
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
G__inference_conv2d_658_layer_call_and_return_conditional_losses_4209802╣
"conv2d_655/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_109/StatefulPartitionedCall:output:0conv2d_655_4209820conv2d_655_4209822*
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
G__inference_conv2d_655_layer_call_and_return_conditional_losses_4209819м
"conv2d_659/StatefulPartitionedCallStatefulPartitionedCall+conv2d_658/StatefulPartitionedCall:output:0conv2d_659_4209837conv2d_659_4209839*
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
G__inference_conv2d_659_layer_call_and_return_conditional_losses_4209836м
"conv2d_656/StatefulPartitionedCallStatefulPartitionedCall+conv2d_655/StatefulPartitionedCall:output:0conv2d_656_4209854conv2d_656_4209856*
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
G__inference_conv2d_656_layer_call_and_return_conditional_losses_4209853Б
(global_max_pooling2d_109/PartitionedCallPartitionedCall+conv2d_659/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *^
fYRW
U__inference_global_max_pooling2d_109_layer_call_and_return_conditional_losses_4209735э
flatten_219/PartitionedCallPartitionedCall1global_max_pooling2d_109/PartitionedCall:output:0*
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
H__inference_flatten_219_layer_call_and_return_conditional_losses_4209866ш
flatten_218/PartitionedCallPartitionedCall+conv2d_656/StatefulPartitionedCall:output:0*
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
H__inference_flatten_218_layer_call_and_return_conditional_losses_4209874Р
concatenate_109/PartitionedCallPartitionedCall$flatten_219/PartitionedCall:output:0$flatten_218/PartitionedCall:output:0*
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
GPU2*0J 8В *U
fPRN
L__inference_concatenate_109_layer_call_and_return_conditional_losses_4209883Ь
!dense_327/StatefulPartitionedCallStatefulPartitionedCall(concatenate_109/PartitionedCall:output:0dense_327_4209897dense_327_4209899*
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
F__inference_dense_327_layer_call_and_return_conditional_losses_4209896Ю
!dense_328/StatefulPartitionedCallStatefulPartitionedCall*dense_327/StatefulPartitionedCall:output:0dense_328_4209914dense_328_4209916*
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
F__inference_dense_328_layer_call_and_return_conditional_losses_4209913Ю
!dense_329/StatefulPartitionedCallStatefulPartitionedCall*dense_328/StatefulPartitionedCall:output:0dense_329_4209931dense_329_4209933*
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
F__inference_dense_329_layer_call_and_return_conditional_losses_4209930╞
+prediction_output_0/StatefulPartitionedCallStatefulPartitionedCall*dense_329/StatefulPartitionedCall:output:0prediction_output_0_4209947prediction_output_0_4209949*
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
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4209946Г
IdentityIdentity4prediction_output_0/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Ё
NoOpNoOp0^batch_normalization_109/StatefulPartitionedCall#^conv2d_654/StatefulPartitionedCall#^conv2d_655/StatefulPartitionedCall#^conv2d_656/StatefulPartitionedCall#^conv2d_657/StatefulPartitionedCall#^conv2d_658/StatefulPartitionedCall#^conv2d_659/StatefulPartitionedCall"^dense_327/StatefulPartitionedCall"^dense_328/StatefulPartitionedCall"^dense_329/StatefulPartitionedCall,^prediction_output_0/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:         Й:         ЙЙ: : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_109/StatefulPartitionedCall/batch_normalization_109/StatefulPartitionedCall2H
"conv2d_654/StatefulPartitionedCall"conv2d_654/StatefulPartitionedCall2H
"conv2d_655/StatefulPartitionedCall"conv2d_655/StatefulPartitionedCall2H
"conv2d_656/StatefulPartitionedCall"conv2d_656/StatefulPartitionedCall2H
"conv2d_657/StatefulPartitionedCall"conv2d_657/StatefulPartitionedCall2H
"conv2d_658/StatefulPartitionedCall"conv2d_658/StatefulPartitionedCall2H
"conv2d_659/StatefulPartitionedCall"conv2d_659/StatefulPartitionedCall2F
!dense_327/StatefulPartitionedCall!dense_327/StatefulPartitionedCall2F
!dense_328/StatefulPartitionedCall!dense_328/StatefulPartitionedCall2F
!dense_329/StatefulPartitionedCall!dense_329/StatefulPartitionedCall2Z
+prediction_output_0/StatefulPartitionedCall+prediction_output_0/StatefulPartitionedCall:X T
0
_output_shapes
:         Й
 
_user_specified_nameinputs:YU
1
_output_shapes
:         ЙЙ
 
_user_specified_nameinputs
╙	
Б
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4209946

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
ё
p
R__inference_spatial_dropout2d_109_layer_call_and_return_conditional_losses_4209630

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
┴
v
L__inference_concatenate_109_layer_call_and_return_conditional_losses_4209883

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
▌
┤
-__inference_joint_model_layer_call_fn_4210004
	input_219
	input_220!
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
StatefulPartitionedCallStatefulPartitionedCall	input_219	input_220unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
H__inference_joint_model_layer_call_and_return_conditional_losses_4209953o
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
_user_specified_name	input_219:\X
1
_output_shapes
:         ЙЙ
#
_user_specified_name	input_220
Й
├
T__inference_batch_normalization_109_layer_call_and_return_conditional_losses_4211034

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
пЩ
ь
H__inference_joint_model_layer_call_and_return_conditional_losses_4210874
inputs_0
inputs_1C
)conv2d_657_conv2d_readvariableop_resource:		8
*conv2d_657_biasadd_readvariableop_resource:C
)conv2d_654_conv2d_readvariableop_resource:8
*conv2d_654_biasadd_readvariableop_resource:=
/batch_normalization_109_readvariableop_resource:?
1batch_normalization_109_readvariableop_1_resource:N
@batch_normalization_109_fusedbatchnormv3_readvariableop_resource:P
Bbatch_normalization_109_fusedbatchnormv3_readvariableop_1_resource:C
)conv2d_658_conv2d_readvariableop_resource:8
*conv2d_658_biasadd_readvariableop_resource:C
)conv2d_655_conv2d_readvariableop_resource:	8
*conv2d_655_biasadd_readvariableop_resource:C
)conv2d_659_conv2d_readvariableop_resource:8
*conv2d_659_biasadd_readvariableop_resource:C
)conv2d_656_conv2d_readvariableop_resource:8
*conv2d_656_biasadd_readvariableop_resource:;
(dense_327_matmul_readvariableop_resource:	Ф7
)dense_327_biasadd_readvariableop_resource::
(dense_328_matmul_readvariableop_resource:7
)dense_328_biasadd_readvariableop_resource::
(dense_329_matmul_readvariableop_resource:7
)dense_329_biasadd_readvariableop_resource:D
2prediction_output_0_matmul_readvariableop_resource:A
3prediction_output_0_biasadd_readvariableop_resource:
identityИв&batch_normalization_109/AssignNewValueв(batch_normalization_109/AssignNewValue_1в7batch_normalization_109/FusedBatchNormV3/ReadVariableOpв9batch_normalization_109/FusedBatchNormV3/ReadVariableOp_1в&batch_normalization_109/ReadVariableOpв(batch_normalization_109/ReadVariableOp_1в!conv2d_654/BiasAdd/ReadVariableOpв conv2d_654/Conv2D/ReadVariableOpв!conv2d_655/BiasAdd/ReadVariableOpв conv2d_655/Conv2D/ReadVariableOpв!conv2d_656/BiasAdd/ReadVariableOpв conv2d_656/Conv2D/ReadVariableOpв!conv2d_657/BiasAdd/ReadVariableOpв conv2d_657/Conv2D/ReadVariableOpв!conv2d_658/BiasAdd/ReadVariableOpв conv2d_658/Conv2D/ReadVariableOpв!conv2d_659/BiasAdd/ReadVariableOpв conv2d_659/Conv2D/ReadVariableOpв dense_327/BiasAdd/ReadVariableOpвdense_327/MatMul/ReadVariableOpв dense_328/BiasAdd/ReadVariableOpвdense_328/MatMul/ReadVariableOpв dense_329/BiasAdd/ReadVariableOpвdense_329/MatMul/ReadVariableOpв*prediction_output_0/BiasAdd/ReadVariableOpв)prediction_output_0/MatMul/ReadVariableOpТ
 conv2d_657/Conv2D/ReadVariableOpReadVariableOp)conv2d_657_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0│
conv2d_657/Conv2DConv2Dinputs_1(conv2d_657/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides	
ЙИ
!conv2d_657/BiasAdd/ReadVariableOpReadVariableOp*conv2d_657_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv2d_657/BiasAddBiasAddconv2d_657/Conv2D:output:0)conv2d_657/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Йo
conv2d_657/ReluReluconv2d_657/BiasAdd:output:0*
T0*0
_output_shapes
:         ЙТ
 conv2d_654/Conv2D/ReadVariableOpReadVariableOp)conv2d_654_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0▓
conv2d_654/Conv2DConv2Dinputs_0(conv2d_654/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides
И
!conv2d_654/BiasAdd/ReadVariableOpReadVariableOp*conv2d_654_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv2d_654/BiasAddBiasAddconv2d_654/Conv2D:output:0)conv2d_654/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Йo
conv2d_654/ReluReluconv2d_654/BiasAdd:output:0*
T0*0
_output_shapes
:         Йh
spatial_dropout2d_109/ShapeShapeconv2d_657/Relu:activations:0*
T0*
_output_shapes
:s
)spatial_dropout2d_109/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+spatial_dropout2d_109/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+spatial_dropout2d_109/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#spatial_dropout2d_109/strided_sliceStridedSlice$spatial_dropout2d_109/Shape:output:02spatial_dropout2d_109/strided_slice/stack:output:04spatial_dropout2d_109/strided_slice/stack_1:output:04spatial_dropout2d_109/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
+spatial_dropout2d_109/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-spatial_dropout2d_109/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-spatial_dropout2d_109/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╟
%spatial_dropout2d_109/strided_slice_1StridedSlice$spatial_dropout2d_109/Shape:output:04spatial_dropout2d_109/strided_slice_1/stack:output:06spatial_dropout2d_109/strided_slice_1/stack_1:output:06spatial_dropout2d_109/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
#spatial_dropout2d_109/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?░
!spatial_dropout2d_109/dropout/MulMulconv2d_657/Relu:activations:0,spatial_dropout2d_109/dropout/Const:output:0*
T0*0
_output_shapes
:         Йv
4spatial_dropout2d_109/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :v
4spatial_dropout2d_109/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :─
2spatial_dropout2d_109/dropout/random_uniform/shapePack,spatial_dropout2d_109/strided_slice:output:0=spatial_dropout2d_109/dropout/random_uniform/shape/1:output:0=spatial_dropout2d_109/dropout/random_uniform/shape/2:output:0.spatial_dropout2d_109/strided_slice_1:output:0*
N*
T0*
_output_shapes
:╧
:spatial_dropout2d_109/dropout/random_uniform/RandomUniformRandomUniform;spatial_dropout2d_109/dropout/random_uniform/shape:output:0*
T0*/
_output_shapes
:         *
dtype0q
,spatial_dropout2d_109/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Ё
*spatial_dropout2d_109/dropout/GreaterEqualGreaterEqualCspatial_dropout2d_109/dropout/random_uniform/RandomUniform:output:05spatial_dropout2d_109/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         г
"spatial_dropout2d_109/dropout/CastCast.spatial_dropout2d_109/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         ┤
#spatial_dropout2d_109/dropout/Mul_1Mul%spatial_dropout2d_109/dropout/Mul:z:0&spatial_dropout2d_109/dropout/Cast:y:0*
T0*0
_output_shapes
:         ЙТ
&batch_normalization_109/ReadVariableOpReadVariableOp/batch_normalization_109_readvariableop_resource*
_output_shapes
:*
dtype0Ц
(batch_normalization_109/ReadVariableOp_1ReadVariableOp1batch_normalization_109_readvariableop_1_resource*
_output_shapes
:*
dtype0┤
7batch_normalization_109/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_109_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╕
9batch_normalization_109/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_109_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╘
(batch_normalization_109/FusedBatchNormV3FusedBatchNormV3conv2d_654/Relu:activations:0.batch_normalization_109/ReadVariableOp:value:00batch_normalization_109/ReadVariableOp_1:value:0?batch_normalization_109/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_109/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         Й:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<ж
&batch_normalization_109/AssignNewValueAssignVariableOp@batch_normalization_109_fusedbatchnormv3_readvariableop_resource5batch_normalization_109/FusedBatchNormV3:batch_mean:08^batch_normalization_109/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(░
(batch_normalization_109/AssignNewValue_1AssignVariableOpBbatch_normalization_109_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_109/FusedBatchNormV3:batch_variance:0:^batch_normalization_109/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Т
 conv2d_658/Conv2D/ReadVariableOpReadVariableOp)conv2d_658_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╥
conv2d_658/Conv2DConv2D'spatial_dropout2d_109/dropout/Mul_1:z:0(conv2d_658/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides	
ЙИ
!conv2d_658/BiasAdd/ReadVariableOpReadVariableOp*conv2d_658_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv2d_658/BiasAddBiasAddconv2d_658/Conv2D:output:0)conv2d_658/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Йo
conv2d_658/ReluReluconv2d_658/BiasAdd:output:0*
T0*0
_output_shapes
:         ЙТ
 conv2d_655/Conv2D/ReadVariableOpReadVariableOp)conv2d_655_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0╓
conv2d_655/Conv2DConv2D,batch_normalization_109/FusedBatchNormV3:y:0(conv2d_655/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides
И
!conv2d_655/BiasAdd/ReadVariableOpReadVariableOp*conv2d_655_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv2d_655/BiasAddBiasAddconv2d_655/Conv2D:output:0)conv2d_655/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Йo
conv2d_655/ReluReluconv2d_655/BiasAdd:output:0*
T0*0
_output_shapes
:         ЙТ
 conv2d_659/Conv2D/ReadVariableOpReadVariableOp)conv2d_659_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╚
conv2d_659/Conv2DConv2Dconv2d_658/Relu:activations:0(conv2d_659/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides	
ЙИ
!conv2d_659/BiasAdd/ReadVariableOpReadVariableOp*conv2d_659_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv2d_659/BiasAddBiasAddconv2d_659/Conv2D:output:0)conv2d_659/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Йo
conv2d_659/ReluReluconv2d_659/BiasAdd:output:0*
T0*0
_output_shapes
:         ЙТ
 conv2d_656/Conv2D/ReadVariableOpReadVariableOp)conv2d_656_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╟
conv2d_656/Conv2DConv2Dconv2d_655/Relu:activations:0(conv2d_656/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Й*
paddingSAME*
strides
И
!conv2d_656/BiasAdd/ReadVariableOpReadVariableOp*conv2d_656_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv2d_656/BiasAddBiasAddconv2d_656/Conv2D:output:0)conv2d_656/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Йo
conv2d_656/ReluReluconv2d_656/BiasAdd:output:0*
T0*0
_output_shapes
:         Й
.global_max_pooling2d_109/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      н
global_max_pooling2d_109/MaxMaxconv2d_659/Relu:activations:07global_max_pooling2d_109/Max/reduction_indices:output:0*
T0*'
_output_shapes
:         b
flatten_219/ConstConst*
_output_shapes
:*
dtype0*
valueB"       У
flatten_219/ReshapeReshape%global_max_pooling2d_109/Max:output:0flatten_219/Const:output:0*
T0*'
_output_shapes
:         b
flatten_218/ConstConst*
_output_shapes
:*
dtype0*
valueB"      М
flatten_218/ReshapeReshapeconv2d_656/Relu:activations:0flatten_218/Const:output:0*
T0*(
_output_shapes
:         Т]
concatenate_109/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :└
concatenate_109/concatConcatV2flatten_219/Reshape:output:0flatten_218/Reshape:output:0$concatenate_109/concat/axis:output:0*
N*
T0*(
_output_shapes
:         ФЙ
dense_327/MatMul/ReadVariableOpReadVariableOp(dense_327_matmul_readvariableop_resource*
_output_shapes
:	Ф*
dtype0Ц
dense_327/MatMulMatMulconcatenate_109/concat:output:0'dense_327/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ж
 dense_327/BiasAdd/ReadVariableOpReadVariableOp)dense_327_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
dense_327/BiasAddBiasAdddense_327/MatMul:product:0(dense_327/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d
dense_327/ReluReludense_327/BiasAdd:output:0*
T0*'
_output_shapes
:         И
dense_328/MatMul/ReadVariableOpReadVariableOp(dense_328_matmul_readvariableop_resource*
_output_shapes

:*
dtype0У
dense_328/MatMulMatMuldense_327/Relu:activations:0'dense_328/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ж
 dense_328/BiasAdd/ReadVariableOpReadVariableOp)dense_328_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
dense_328/BiasAddBiasAdddense_328/MatMul:product:0(dense_328/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d
dense_328/ReluReludense_328/BiasAdd:output:0*
T0*'
_output_shapes
:         И
dense_329/MatMul/ReadVariableOpReadVariableOp(dense_329_matmul_readvariableop_resource*
_output_shapes

:*
dtype0У
dense_329/MatMulMatMuldense_328/Relu:activations:0'dense_329/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ж
 dense_329/BiasAdd/ReadVariableOpReadVariableOp)dense_329_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
dense_329/BiasAddBiasAdddense_329/MatMul:product:0(dense_329/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d
dense_329/ReluReludense_329/BiasAdd:output:0*
T0*'
_output_shapes
:         Ь
)prediction_output_0/MatMul/ReadVariableOpReadVariableOp2prediction_output_0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0з
prediction_output_0/MatMulMatMuldense_329/Relu:activations:01prediction_output_0/MatMul/ReadVariableOp:value:0*
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
:         ╢
NoOpNoOp'^batch_normalization_109/AssignNewValue)^batch_normalization_109/AssignNewValue_18^batch_normalization_109/FusedBatchNormV3/ReadVariableOp:^batch_normalization_109/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_109/ReadVariableOp)^batch_normalization_109/ReadVariableOp_1"^conv2d_654/BiasAdd/ReadVariableOp!^conv2d_654/Conv2D/ReadVariableOp"^conv2d_655/BiasAdd/ReadVariableOp!^conv2d_655/Conv2D/ReadVariableOp"^conv2d_656/BiasAdd/ReadVariableOp!^conv2d_656/Conv2D/ReadVariableOp"^conv2d_657/BiasAdd/ReadVariableOp!^conv2d_657/Conv2D/ReadVariableOp"^conv2d_658/BiasAdd/ReadVariableOp!^conv2d_658/Conv2D/ReadVariableOp"^conv2d_659/BiasAdd/ReadVariableOp!^conv2d_659/Conv2D/ReadVariableOp!^dense_327/BiasAdd/ReadVariableOp ^dense_327/MatMul/ReadVariableOp!^dense_328/BiasAdd/ReadVariableOp ^dense_328/MatMul/ReadVariableOp!^dense_329/BiasAdd/ReadVariableOp ^dense_329/MatMul/ReadVariableOp+^prediction_output_0/BiasAdd/ReadVariableOp*^prediction_output_0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:         Й:         ЙЙ: : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_109/AssignNewValue&batch_normalization_109/AssignNewValue2T
(batch_normalization_109/AssignNewValue_1(batch_normalization_109/AssignNewValue_12r
7batch_normalization_109/FusedBatchNormV3/ReadVariableOp7batch_normalization_109/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_109/FusedBatchNormV3/ReadVariableOp_19batch_normalization_109/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_109/ReadVariableOp&batch_normalization_109/ReadVariableOp2T
(batch_normalization_109/ReadVariableOp_1(batch_normalization_109/ReadVariableOp_12F
!conv2d_654/BiasAdd/ReadVariableOp!conv2d_654/BiasAdd/ReadVariableOp2D
 conv2d_654/Conv2D/ReadVariableOp conv2d_654/Conv2D/ReadVariableOp2F
!conv2d_655/BiasAdd/ReadVariableOp!conv2d_655/BiasAdd/ReadVariableOp2D
 conv2d_655/Conv2D/ReadVariableOp conv2d_655/Conv2D/ReadVariableOp2F
!conv2d_656/BiasAdd/ReadVariableOp!conv2d_656/BiasAdd/ReadVariableOp2D
 conv2d_656/Conv2D/ReadVariableOp conv2d_656/Conv2D/ReadVariableOp2F
!conv2d_657/BiasAdd/ReadVariableOp!conv2d_657/BiasAdd/ReadVariableOp2D
 conv2d_657/Conv2D/ReadVariableOp conv2d_657/Conv2D/ReadVariableOp2F
!conv2d_658/BiasAdd/ReadVariableOp!conv2d_658/BiasAdd/ReadVariableOp2D
 conv2d_658/Conv2D/ReadVariableOp conv2d_658/Conv2D/ReadVariableOp2F
!conv2d_659/BiasAdd/ReadVariableOp!conv2d_659/BiasAdd/ReadVariableOp2D
 conv2d_659/Conv2D/ReadVariableOp conv2d_659/Conv2D/ReadVariableOp2D
 dense_327/BiasAdd/ReadVariableOp dense_327/BiasAdd/ReadVariableOp2B
dense_327/MatMul/ReadVariableOpdense_327/MatMul/ReadVariableOp2D
 dense_328/BiasAdd/ReadVariableOp dense_328/BiasAdd/ReadVariableOp2B
dense_328/MatMul/ReadVariableOpdense_328/MatMul/ReadVariableOp2D
 dense_329/BiasAdd/ReadVariableOp dense_329/BiasAdd/ReadVariableOp2B
dense_329/MatMul/ReadVariableOpdense_329/MatMul/ReadVariableOp2X
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
Й
├
T__inference_batch_normalization_109_layer_call_and_return_conditional_losses_4209714

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
Э

ў
F__inference_dense_328_layer_call_and_return_conditional_losses_4209913

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
╔
Ш
+__inference_dense_329_layer_call_fn_4211189

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
F__inference_dense_329_layer_call_and_return_conditional_losses_4209930o
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
Ш	
╘
9__inference_batch_normalization_109_layer_call_fn_4210998

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallЮ
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
GPU2*0J 8В *]
fXRV
T__inference_batch_normalization_109_layer_call_and_return_conditional_losses_4209714Й
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
Ьо
▌0
#__inference__traced_restore_4211703
file_prefix<
"assignvariableop_conv2d_657_kernel:		0
"assignvariableop_1_conv2d_657_bias:>
$assignvariableop_2_conv2d_654_kernel:0
"assignvariableop_3_conv2d_654_bias:>
$assignvariableop_4_conv2d_658_kernel:0
"assignvariableop_5_conv2d_658_bias:>
0assignvariableop_6_batch_normalization_109_gamma:=
/assignvariableop_7_batch_normalization_109_beta:D
6assignvariableop_8_batch_normalization_109_moving_mean:H
:assignvariableop_9_batch_normalization_109_moving_variance:?
%assignvariableop_10_conv2d_659_kernel:1
#assignvariableop_11_conv2d_659_bias:?
%assignvariableop_12_conv2d_655_kernel:	1
#assignvariableop_13_conv2d_655_bias:?
%assignvariableop_14_conv2d_656_kernel:1
#assignvariableop_15_conv2d_656_bias:7
$assignvariableop_16_dense_327_kernel:	Ф0
"assignvariableop_17_dense_327_bias:6
$assignvariableop_18_dense_328_kernel:0
"assignvariableop_19_dense_328_bias:6
$assignvariableop_20_dense_329_kernel:0
"assignvariableop_21_dense_329_bias:@
.assignvariableop_22_prediction_output_0_kernel::
,assignvariableop_23_prediction_output_0_bias:$
assignvariableop_24_beta_1: $
assignvariableop_25_beta_2: #
assignvariableop_26_decay: +
!assignvariableop_27_learning_rate: '
assignvariableop_28_adam_iter:	 #
assignvariableop_29_total: #
assignvariableop_30_count: F
,assignvariableop_31_adam_conv2d_657_kernel_m:		8
*assignvariableop_32_adam_conv2d_657_bias_m:F
,assignvariableop_33_adam_conv2d_654_kernel_m:8
*assignvariableop_34_adam_conv2d_654_bias_m:F
,assignvariableop_35_adam_conv2d_658_kernel_m:8
*assignvariableop_36_adam_conv2d_658_bias_m:F
8assignvariableop_37_adam_batch_normalization_109_gamma_m:E
7assignvariableop_38_adam_batch_normalization_109_beta_m:F
,assignvariableop_39_adam_conv2d_659_kernel_m:8
*assignvariableop_40_adam_conv2d_659_bias_m:F
,assignvariableop_41_adam_conv2d_655_kernel_m:	8
*assignvariableop_42_adam_conv2d_655_bias_m:F
,assignvariableop_43_adam_conv2d_656_kernel_m:8
*assignvariableop_44_adam_conv2d_656_bias_m:>
+assignvariableop_45_adam_dense_327_kernel_m:	Ф7
)assignvariableop_46_adam_dense_327_bias_m:=
+assignvariableop_47_adam_dense_328_kernel_m:7
)assignvariableop_48_adam_dense_328_bias_m:=
+assignvariableop_49_adam_dense_329_kernel_m:7
)assignvariableop_50_adam_dense_329_bias_m:G
5assignvariableop_51_adam_prediction_output_0_kernel_m:A
3assignvariableop_52_adam_prediction_output_0_bias_m:F
,assignvariableop_53_adam_conv2d_657_kernel_v:		8
*assignvariableop_54_adam_conv2d_657_bias_v:F
,assignvariableop_55_adam_conv2d_654_kernel_v:8
*assignvariableop_56_adam_conv2d_654_bias_v:F
,assignvariableop_57_adam_conv2d_658_kernel_v:8
*assignvariableop_58_adam_conv2d_658_bias_v:F
8assignvariableop_59_adam_batch_normalization_109_gamma_v:E
7assignvariableop_60_adam_batch_normalization_109_beta_v:F
,assignvariableop_61_adam_conv2d_659_kernel_v:8
*assignvariableop_62_adam_conv2d_659_bias_v:F
,assignvariableop_63_adam_conv2d_655_kernel_v:	8
*assignvariableop_64_adam_conv2d_655_bias_v:F
,assignvariableop_65_adam_conv2d_656_kernel_v:8
*assignvariableop_66_adam_conv2d_656_bias_v:>
+assignvariableop_67_adam_dense_327_kernel_v:	Ф7
)assignvariableop_68_adam_dense_327_bias_v:=
+assignvariableop_69_adam_dense_328_kernel_v:7
)assignvariableop_70_adam_dense_328_bias_v:=
+assignvariableop_71_adam_dense_329_kernel_v:7
)assignvariableop_72_adam_dense_329_bias_v:G
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
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_657_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_657_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_2AssignVariableOp$assignvariableop_2_conv2d_654_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_3AssignVariableOp"assignvariableop_3_conv2d_654_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_4AssignVariableOp$assignvariableop_4_conv2d_658_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_5AssignVariableOp"assignvariableop_5_conv2d_658_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_6AssignVariableOp0assignvariableop_6_batch_normalization_109_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_7AssignVariableOp/assignvariableop_7_batch_normalization_109_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_8AssignVariableOp6assignvariableop_8_batch_normalization_109_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_9AssignVariableOp:assignvariableop_9_batch_normalization_109_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_10AssignVariableOp%assignvariableop_10_conv2d_659_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_11AssignVariableOp#assignvariableop_11_conv2d_659_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_12AssignVariableOp%assignvariableop_12_conv2d_655_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_13AssignVariableOp#assignvariableop_13_conv2d_655_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_14AssignVariableOp%assignvariableop_14_conv2d_656_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_15AssignVariableOp#assignvariableop_15_conv2d_656_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_16AssignVariableOp$assignvariableop_16_dense_327_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_17AssignVariableOp"assignvariableop_17_dense_327_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_18AssignVariableOp$assignvariableop_18_dense_328_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_19AssignVariableOp"assignvariableop_19_dense_328_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_20AssignVariableOp$assignvariableop_20_dense_329_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_21AssignVariableOp"assignvariableop_21_dense_329_biasIdentity_21:output:0"/device:CPU:0*
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
AssignVariableOp_31AssignVariableOp,assignvariableop_31_adam_conv2d_657_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_conv2d_657_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_33AssignVariableOp,assignvariableop_33_adam_conv2d_654_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_conv2d_654_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_35AssignVariableOp,assignvariableop_35_adam_conv2d_658_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_36AssignVariableOp*assignvariableop_36_adam_conv2d_658_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_37AssignVariableOp8assignvariableop_37_adam_batch_normalization_109_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_38AssignVariableOp7assignvariableop_38_adam_batch_normalization_109_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_39AssignVariableOp,assignvariableop_39_adam_conv2d_659_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_conv2d_659_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_41AssignVariableOp,assignvariableop_41_adam_conv2d_655_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_conv2d_655_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_43AssignVariableOp,assignvariableop_43_adam_conv2d_656_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_conv2d_656_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_dense_327_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_dense_327_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_dense_328_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_dense_328_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_dense_329_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_dense_329_bias_mIdentity_50:output:0"/device:CPU:0*
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
AssignVariableOp_53AssignVariableOp,assignvariableop_53_adam_conv2d_657_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_54AssignVariableOp*assignvariableop_54_adam_conv2d_657_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_55AssignVariableOp,assignvariableop_55_adam_conv2d_654_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_56AssignVariableOp*assignvariableop_56_adam_conv2d_654_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_57AssignVariableOp,assignvariableop_57_adam_conv2d_658_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_conv2d_658_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_59AssignVariableOp8assignvariableop_59_adam_batch_normalization_109_gamma_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_60AssignVariableOp7assignvariableop_60_adam_batch_normalization_109_beta_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_61AssignVariableOp,assignvariableop_61_adam_conv2d_659_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_62AssignVariableOp*assignvariableop_62_adam_conv2d_659_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_63AssignVariableOp,assignvariableop_63_adam_conv2d_655_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_64AssignVariableOp*assignvariableop_64_adam_conv2d_655_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_65AssignVariableOp,assignvariableop_65_adam_conv2d_656_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_66AssignVariableOp*assignvariableop_66_adam_conv2d_656_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_dense_327_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_dense_327_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_dense_328_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_dense_328_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_dense_329_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_dense_329_bias_vIdentity_72:output:0"/device:CPU:0*
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
Э

ў
F__inference_dense_329_layer_call_and_return_conditional_losses_4211200

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
Н
А
G__inference_conv2d_658_layer_call_and_return_conditional_losses_4209802

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
╫
▓
-__inference_joint_model_layer_call_fn_4210608
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
H__inference_joint_model_layer_call_and_return_conditional_losses_4209953o
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
П
А
G__inference_conv2d_657_layer_call_and_return_conditional_losses_4210894

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
Ы
p
7__inference_spatial_dropout2d_109_layer_call_fn_4210904

inputs
identityИвStatefulPartitionedCallє
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
GPU2*0J 8В *[
fVRT
R__inference_spatial_dropout2d_109_layer_call_and_return_conditional_losses_4209658Т
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
Э

ў
F__inference_dense_329_layer_call_and_return_conditional_losses_4209930

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
ў
б
,__inference_conv2d_655_layer_call_fn_4211063

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
G__inference_conv2d_655_layer_call_and_return_conditional_losses_4209819x
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
 
_user_specified_nameinputs"╡	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*О
serving_default·
H
	input_219;
serving_default_input_219:0         Й
I
	input_220<
serving_default_input_220:0         ЙЙG
prediction_output_00
StatefulPartitionedCall:0         tensorflow/serving/predict:■Й
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
-__inference_joint_model_layer_call_fn_4210004
-__inference_joint_model_layer_call_fn_4210608
-__inference_joint_model_layer_call_fn_4210662
-__inference_joint_model_layer_call_fn_4210354┐
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
H__inference_joint_model_layer_call_and_return_conditional_losses_4210759
H__inference_joint_model_layer_call_and_return_conditional_losses_4210874
H__inference_joint_model_layer_call_and_return_conditional_losses_4210423
H__inference_joint_model_layer_call_and_return_conditional_losses_4210492┐
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
"__inference__wrapped_model_4209621	input_219	input_220"Ш
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
,__inference_conv2d_657_layer_call_fn_4210883в
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
G__inference_conv2d_657_layer_call_and_return_conditional_losses_4210894в
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
+:)		2conv2d_657/kernel
:2conv2d_657/bias
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
у
╗trace_0
╝trace_12и
7__inference_spatial_dropout2d_109_layer_call_fn_4210899
7__inference_spatial_dropout2d_109_layer_call_fn_4210904│
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
Щ
╜trace_0
╛trace_12▐
R__inference_spatial_dropout2d_109_layer_call_and_return_conditional_losses_4210909
R__inference_spatial_dropout2d_109_layer_call_and_return_conditional_losses_4210932│
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
,__inference_conv2d_654_layer_call_fn_4210941в
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
G__inference_conv2d_654_layer_call_and_return_conditional_losses_4210952в
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
+:)2conv2d_654/kernel
:2conv2d_654/bias
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
,__inference_conv2d_658_layer_call_fn_4210961в
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
G__inference_conv2d_658_layer_call_and_return_conditional_losses_4210972в
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
+:)2conv2d_658/kernel
:2conv2d_658/bias
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
ч
╥trace_0
╙trace_12м
9__inference_batch_normalization_109_layer_call_fn_4210985
9__inference_batch_normalization_109_layer_call_fn_4210998│
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
Э
╘trace_0
╒trace_12т
T__inference_batch_normalization_109_layer_call_and_return_conditional_losses_4211016
T__inference_batch_normalization_109_layer_call_and_return_conditional_losses_4211034│
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
+:)2batch_normalization_109/gamma
*:(2batch_normalization_109/beta
3:1 (2#batch_normalization_109/moving_mean
7:5 (2'batch_normalization_109/moving_variance
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
,__inference_conv2d_659_layer_call_fn_4211043в
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
G__inference_conv2d_659_layer_call_and_return_conditional_losses_4211054в
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
+:)2conv2d_659/kernel
:2conv2d_659/bias
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
,__inference_conv2d_655_layer_call_fn_4211063в
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
G__inference_conv2d_655_layer_call_and_return_conditional_losses_4211074в
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
+:)	2conv2d_655/kernel
:2conv2d_655/bias
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
А
щtrace_02с
:__inference_global_max_pooling2d_109_layer_call_fn_4211079в
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
Ы
ъtrace_02№
U__inference_global_max_pooling2d_109_layer_call_and_return_conditional_losses_4211085в
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
,__inference_conv2d_656_layer_call_fn_4211094в
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
G__inference_conv2d_656_layer_call_and_return_conditional_losses_4211105в
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
+:)2conv2d_656/kernel
:2conv2d_656/bias
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
-__inference_flatten_219_layer_call_fn_4211110в
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
H__inference_flatten_219_layer_call_and_return_conditional_losses_4211116в
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
-__inference_flatten_218_layer_call_fn_4211121в
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
H__inference_flatten_218_layer_call_and_return_conditional_losses_4211127в
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
ў
Еtrace_02╪
1__inference_concatenate_109_layer_call_fn_4211133в
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
Т
Жtrace_02є
L__inference_concatenate_109_layer_call_and_return_conditional_losses_4211140в
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
+__inference_dense_327_layer_call_fn_4211149в
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
F__inference_dense_327_layer_call_and_return_conditional_losses_4211160в
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
#:!	Ф2dense_327/kernel
:2dense_327/bias
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
+__inference_dense_328_layer_call_fn_4211169в
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
F__inference_dense_328_layer_call_and_return_conditional_losses_4211180в
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
": 2dense_328/kernel
:2dense_328/bias
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
+__inference_dense_329_layer_call_fn_4211189в
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
F__inference_dense_329_layer_call_and_return_conditional_losses_4211200в
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
": 2dense_329/kernel
:2dense_329/bias
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
5__inference_prediction_output_0_layer_call_fn_4211209в
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
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4211219в
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
-__inference_joint_model_layer_call_fn_4210004	input_219	input_220"┐
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
-__inference_joint_model_layer_call_fn_4210608inputs/0inputs/1"┐
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
-__inference_joint_model_layer_call_fn_4210662inputs/0inputs/1"┐
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
-__inference_joint_model_layer_call_fn_4210354	input_219	input_220"┐
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
H__inference_joint_model_layer_call_and_return_conditional_losses_4210759inputs/0inputs/1"┐
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
H__inference_joint_model_layer_call_and_return_conditional_losses_4210874inputs/0inputs/1"┐
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
H__inference_joint_model_layer_call_and_return_conditional_losses_4210423	input_219	input_220"┐
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
H__inference_joint_model_layer_call_and_return_conditional_losses_4210492	input_219	input_220"┐
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
%__inference_signature_wrapper_4210554	input_219	input_220"Ф
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
,__inference_conv2d_657_layer_call_fn_4210883inputs"в
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
G__inference_conv2d_657_layer_call_and_return_conditional_losses_4210894inputs"в
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
№B∙
7__inference_spatial_dropout2d_109_layer_call_fn_4210899inputs"│
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
№B∙
7__inference_spatial_dropout2d_109_layer_call_fn_4210904inputs"│
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
ЧBФ
R__inference_spatial_dropout2d_109_layer_call_and_return_conditional_losses_4210909inputs"│
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
ЧBФ
R__inference_spatial_dropout2d_109_layer_call_and_return_conditional_losses_4210932inputs"│
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
,__inference_conv2d_654_layer_call_fn_4210941inputs"в
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
G__inference_conv2d_654_layer_call_and_return_conditional_losses_4210952inputs"в
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
,__inference_conv2d_658_layer_call_fn_4210961inputs"в
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
G__inference_conv2d_658_layer_call_and_return_conditional_losses_4210972inputs"в
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
■B√
9__inference_batch_normalization_109_layer_call_fn_4210985inputs"│
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
■B√
9__inference_batch_normalization_109_layer_call_fn_4210998inputs"│
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
ЩBЦ
T__inference_batch_normalization_109_layer_call_and_return_conditional_losses_4211016inputs"│
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
ЩBЦ
T__inference_batch_normalization_109_layer_call_and_return_conditional_losses_4211034inputs"│
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
,__inference_conv2d_659_layer_call_fn_4211043inputs"в
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
G__inference_conv2d_659_layer_call_and_return_conditional_losses_4211054inputs"в
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
,__inference_conv2d_655_layer_call_fn_4211063inputs"в
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
G__inference_conv2d_655_layer_call_and_return_conditional_losses_4211074inputs"в
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
юBы
:__inference_global_max_pooling2d_109_layer_call_fn_4211079inputs"в
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
ЙBЖ
U__inference_global_max_pooling2d_109_layer_call_and_return_conditional_losses_4211085inputs"в
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
,__inference_conv2d_656_layer_call_fn_4211094inputs"в
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
G__inference_conv2d_656_layer_call_and_return_conditional_losses_4211105inputs"в
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
-__inference_flatten_219_layer_call_fn_4211110inputs"в
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
H__inference_flatten_219_layer_call_and_return_conditional_losses_4211116inputs"в
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
-__inference_flatten_218_layer_call_fn_4211121inputs"в
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
H__inference_flatten_218_layer_call_and_return_conditional_losses_4211127inputs"в
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
ёBю
1__inference_concatenate_109_layer_call_fn_4211133inputs/0inputs/1"в
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
МBЙ
L__inference_concatenate_109_layer_call_and_return_conditional_losses_4211140inputs/0inputs/1"в
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
+__inference_dense_327_layer_call_fn_4211149inputs"в
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
F__inference_dense_327_layer_call_and_return_conditional_losses_4211160inputs"в
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
+__inference_dense_328_layer_call_fn_4211169inputs"в
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
F__inference_dense_328_layer_call_and_return_conditional_losses_4211180inputs"в
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
+__inference_dense_329_layer_call_fn_4211189inputs"в
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
F__inference_dense_329_layer_call_and_return_conditional_losses_4211200inputs"в
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
5__inference_prediction_output_0_layer_call_fn_4211209inputs"в
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
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4211219inputs"в
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
0:.		2Adam/conv2d_657/kernel/m
": 2Adam/conv2d_657/bias/m
0:.2Adam/conv2d_654/kernel/m
": 2Adam/conv2d_654/bias/m
0:.2Adam/conv2d_658/kernel/m
": 2Adam/conv2d_658/bias/m
0:.2$Adam/batch_normalization_109/gamma/m
/:-2#Adam/batch_normalization_109/beta/m
0:.2Adam/conv2d_659/kernel/m
": 2Adam/conv2d_659/bias/m
0:.	2Adam/conv2d_655/kernel/m
": 2Adam/conv2d_655/bias/m
0:.2Adam/conv2d_656/kernel/m
": 2Adam/conv2d_656/bias/m
(:&	Ф2Adam/dense_327/kernel/m
!:2Adam/dense_327/bias/m
':%2Adam/dense_328/kernel/m
!:2Adam/dense_328/bias/m
':%2Adam/dense_329/kernel/m
!:2Adam/dense_329/bias/m
1:/2!Adam/prediction_output_0/kernel/m
+:)2Adam/prediction_output_0/bias/m
0:.		2Adam/conv2d_657/kernel/v
": 2Adam/conv2d_657/bias/v
0:.2Adam/conv2d_654/kernel/v
": 2Adam/conv2d_654/bias/v
0:.2Adam/conv2d_658/kernel/v
": 2Adam/conv2d_658/bias/v
0:.2$Adam/batch_normalization_109/gamma/v
/:-2#Adam/batch_normalization_109/beta/v
0:.2Adam/conv2d_659/kernel/v
": 2Adam/conv2d_659/bias/v
0:.	2Adam/conv2d_655/kernel/v
": 2Adam/conv2d_655/bias/v
0:.2Adam/conv2d_656/kernel/v
": 2Adam/conv2d_656/bias/v
(:&	Ф2Adam/dense_327/kernel/v
!:2Adam/dense_327/bias/v
':%2Adam/dense_328/kernel/v
!:2Adam/dense_328/bias/v
':%2Adam/dense_329/kernel/v
!:2Adam/dense_329/bias/v
1:/2!Adam/prediction_output_0/kernel/v
+:)2Adam/prediction_output_0/bias/vЕ
"__inference__wrapped_model_4209621▐ "#23EFGH;<XYOPghВГКЛТУЪЫoвl
eвb
`Ъ]
,К)
	input_219         Й
-К*
	input_220         ЙЙ
к "IкF
D
prediction_output_0-К*
prediction_output_0         я
T__inference_batch_normalization_109_layer_call_and_return_conditional_losses_4211016ЦEFGHMвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ я
T__inference_batch_normalization_109_layer_call_and_return_conditional_losses_4211034ЦEFGHMвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ ╟
9__inference_batch_normalization_109_layer_call_fn_4210985ЙEFGHMвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           ╟
9__inference_batch_normalization_109_layer_call_fn_4210998ЙEFGHMвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           ╓
L__inference_concatenate_109_layer_call_and_return_conditional_losses_4211140Е[вX
QвN
LЪI
"К
inputs/0         
#К 
inputs/1         Т
к "&в#
К
0         Ф
Ъ н
1__inference_concatenate_109_layer_call_fn_4211133x[вX
QвN
LЪI
"К
inputs/0         
#К 
inputs/1         Т
к "К         Ф╣
G__inference_conv2d_654_layer_call_and_return_conditional_losses_4210952n238в5
.в+
)К&
inputs         Й
к ".в+
$К!
0         Й
Ъ С
,__inference_conv2d_654_layer_call_fn_4210941a238в5
.в+
)К&
inputs         Й
к "!К         Й╣
G__inference_conv2d_655_layer_call_and_return_conditional_losses_4211074nXY8в5
.в+
)К&
inputs         Й
к ".в+
$К!
0         Й
Ъ С
,__inference_conv2d_655_layer_call_fn_4211063aXY8в5
.в+
)К&
inputs         Й
к "!К         Й╣
G__inference_conv2d_656_layer_call_and_return_conditional_losses_4211105ngh8в5
.в+
)К&
inputs         Й
к ".в+
$К!
0         Й
Ъ С
,__inference_conv2d_656_layer_call_fn_4211094agh8в5
.в+
)К&
inputs         Й
к "!К         Й║
G__inference_conv2d_657_layer_call_and_return_conditional_losses_4210894o"#9в6
/в,
*К'
inputs         ЙЙ
к ".в+
$К!
0         Й
Ъ Т
,__inference_conv2d_657_layer_call_fn_4210883b"#9в6
/в,
*К'
inputs         ЙЙ
к "!К         Й╣
G__inference_conv2d_658_layer_call_and_return_conditional_losses_4210972n;<8в5
.в+
)К&
inputs         Й
к ".в+
$К!
0         Й
Ъ С
,__inference_conv2d_658_layer_call_fn_4210961a;<8в5
.в+
)К&
inputs         Й
к "!К         Й╣
G__inference_conv2d_659_layer_call_and_return_conditional_losses_4211054nOP8в5
.в+
)К&
inputs         Й
к ".в+
$К!
0         Й
Ъ С
,__inference_conv2d_659_layer_call_fn_4211043aOP8в5
.в+
)К&
inputs         Й
к "!К         Йй
F__inference_dense_327_layer_call_and_return_conditional_losses_4211160_ВГ0в-
&в#
!К
inputs         Ф
к "%в"
К
0         
Ъ Б
+__inference_dense_327_layer_call_fn_4211149RВГ0в-
&в#
!К
inputs         Ф
к "К         и
F__inference_dense_328_layer_call_and_return_conditional_losses_4211180^КЛ/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ А
+__inference_dense_328_layer_call_fn_4211169QКЛ/в,
%в"
 К
inputs         
к "К         и
F__inference_dense_329_layer_call_and_return_conditional_losses_4211200^ТУ/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ А
+__inference_dense_329_layer_call_fn_4211189QТУ/в,
%в"
 К
inputs         
к "К         о
H__inference_flatten_218_layer_call_and_return_conditional_losses_4211127b8в5
.в+
)К&
inputs         Й
к "&в#
К
0         Т
Ъ Ж
-__inference_flatten_218_layer_call_fn_4211121U8в5
.в+
)К&
inputs         Й
к "К         Тд
H__inference_flatten_219_layer_call_and_return_conditional_losses_4211116X/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ |
-__inference_flatten_219_layer_call_fn_4211110K/в,
%в"
 К
inputs         
к "К         ▐
U__inference_global_max_pooling2d_109_layer_call_and_return_conditional_losses_4211085ДRвO
HвE
CК@
inputs4                                    
к ".в+
$К!
0                  
Ъ ╡
:__inference_global_max_pooling2d_109_layer_call_fn_4211079wRвO
HвE
CК@
inputs4                                    
к "!К                  П
H__inference_joint_model_layer_call_and_return_conditional_losses_4210423┬ "#23EFGH;<XYOPghВГКЛТУЪЫwвt
mвj
`Ъ]
,К)
	input_219         Й
-К*
	input_220         ЙЙ
p 

 
к "%в"
К
0         
Ъ П
H__inference_joint_model_layer_call_and_return_conditional_losses_4210492┬ "#23EFGH;<XYOPghВГКЛТУЪЫwвt
mвj
`Ъ]
,К)
	input_219         Й
-К*
	input_220         ЙЙ
p

 
к "%в"
К
0         
Ъ Н
H__inference_joint_model_layer_call_and_return_conditional_losses_4210759└ "#23EFGH;<XYOPghВГКЛТУЪЫuвr
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
H__inference_joint_model_layer_call_and_return_conditional_losses_4210874└ "#23EFGH;<XYOPghВГКЛТУЪЫuвr
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
-__inference_joint_model_layer_call_fn_4210004╡ "#23EFGH;<XYOPghВГКЛТУЪЫwвt
mвj
`Ъ]
,К)
	input_219         Й
-К*
	input_220         ЙЙ
p 

 
к "К         ч
-__inference_joint_model_layer_call_fn_4210354╡ "#23EFGH;<XYOPghВГКЛТУЪЫwвt
mвj
`Ъ]
,К)
	input_219         Й
-К*
	input_220         ЙЙ
p

 
к "К         х
-__inference_joint_model_layer_call_fn_4210608│ "#23EFGH;<XYOPghВГКЛТУЪЫuвr
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
-__inference_joint_model_layer_call_fn_4210662│ "#23EFGH;<XYOPghВГКЛТУЪЫuвr
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
P__inference_prediction_output_0_layer_call_and_return_conditional_losses_4211219^ЪЫ/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ К
5__inference_prediction_output_0_layer_call_fn_4211209QЪЫ/в,
%в"
 К
inputs         
к "К         Я
%__inference_signature_wrapper_4210554ї "#23EFGH;<XYOPghВГКЛТУЪЫЕвБ
в 
zкw
9
	input_219,К)
	input_219         Й
:
	input_220-К*
	input_220         ЙЙ"IкF
D
prediction_output_0-К*
prediction_output_0         ∙
R__inference_spatial_dropout2d_109_layer_call_and_return_conditional_losses_4210909вVвS
LвI
CК@
inputs4                                    
p 
к "HвE
>К;
04                                    
Ъ ∙
R__inference_spatial_dropout2d_109_layer_call_and_return_conditional_losses_4210932вVвS
LвI
CК@
inputs4                                    
p
к "HвE
>К;
04                                    
Ъ ╤
7__inference_spatial_dropout2d_109_layer_call_fn_4210899ХVвS
LвI
CК@
inputs4                                    
p 
к ";К84                                    ╤
7__inference_spatial_dropout2d_109_layer_call_fn_4210904ХVвS
LвI
CК@
inputs4                                    
p
к ";К84                                    
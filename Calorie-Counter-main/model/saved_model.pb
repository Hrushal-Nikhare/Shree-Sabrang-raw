 Å%
¦ü
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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

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
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
>
Maximum
x"T
y"T
z"T"
Ttype:
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
0
Sigmoid
x"T
y"T"
Ttype:

2
-
Sqrt
x"T
y"T"
Ttype:

2
7
Square
x"T
y"T"
Ttype:
2	
Á
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
executor_typestring ¨
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.22v2.8.2-0-g2ea19cbb5758×ê!
z
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_14/kernel
s
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes

:*
dtype0
r
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes
:*
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

block1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock1_conv1/kernel

'block1_conv1/kernel/Read/ReadVariableOpReadVariableOpblock1_conv1/kernel*&
_output_shapes
:@*
dtype0
z
block1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv1/bias
s
%block1_conv1/bias/Read/ReadVariableOpReadVariableOpblock1_conv1/bias*
_output_shapes
:@*
dtype0

block1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameblock1_conv2/kernel

'block1_conv2/kernel/Read/ReadVariableOpReadVariableOpblock1_conv2/kernel*&
_output_shapes
:@@*
dtype0
z
block1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv2/bias
s
%block1_conv2/bias/Read/ReadVariableOpReadVariableOpblock1_conv2/bias*
_output_shapes
:@*
dtype0

block2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock2_conv1/kernel

'block2_conv1/kernel/Read/ReadVariableOpReadVariableOpblock2_conv1/kernel*'
_output_shapes
:@*
dtype0
{
block2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock2_conv1/bias
t
%block2_conv1/bias/Read/ReadVariableOpReadVariableOpblock2_conv1/bias*
_output_shapes	
:*
dtype0

block2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock2_conv2/kernel

'block2_conv2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv2/kernel*(
_output_shapes
:*
dtype0
{
block2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock2_conv2/bias
t
%block2_conv2/bias/Read/ReadVariableOpReadVariableOpblock2_conv2/bias*
_output_shapes	
:*
dtype0

block3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv1/kernel

'block3_conv1/kernel/Read/ReadVariableOpReadVariableOpblock3_conv1/kernel*(
_output_shapes
:*
dtype0
{
block3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv1/bias
t
%block3_conv1/bias/Read/ReadVariableOpReadVariableOpblock3_conv1/bias*
_output_shapes	
:*
dtype0

block3_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv2/kernel

'block3_conv2/kernel/Read/ReadVariableOpReadVariableOpblock3_conv2/kernel*(
_output_shapes
:*
dtype0
{
block3_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv2/bias
t
%block3_conv2/bias/Read/ReadVariableOpReadVariableOpblock3_conv2/bias*
_output_shapes	
:*
dtype0

block3_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv3/kernel

'block3_conv3/kernel/Read/ReadVariableOpReadVariableOpblock3_conv3/kernel*(
_output_shapes
:*
dtype0
{
block3_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv3/bias
t
%block3_conv3/bias/Read/ReadVariableOpReadVariableOpblock3_conv3/bias*
_output_shapes	
:*
dtype0

block3_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv4/kernel

'block3_conv4/kernel/Read/ReadVariableOpReadVariableOpblock3_conv4/kernel*(
_output_shapes
:*
dtype0
{
block3_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv4/bias
t
%block3_conv4/bias/Read/ReadVariableOpReadVariableOpblock3_conv4/bias*
_output_shapes	
:*
dtype0

block4_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv1/kernel

'block4_conv1/kernel/Read/ReadVariableOpReadVariableOpblock4_conv1/kernel*(
_output_shapes
:*
dtype0
{
block4_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv1/bias
t
%block4_conv1/bias/Read/ReadVariableOpReadVariableOpblock4_conv1/bias*
_output_shapes	
:*
dtype0

block4_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv2/kernel

'block4_conv2/kernel/Read/ReadVariableOpReadVariableOpblock4_conv2/kernel*(
_output_shapes
:*
dtype0
{
block4_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv2/bias
t
%block4_conv2/bias/Read/ReadVariableOpReadVariableOpblock4_conv2/bias*
_output_shapes	
:*
dtype0

block4_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv3/kernel

'block4_conv3/kernel/Read/ReadVariableOpReadVariableOpblock4_conv3/kernel*(
_output_shapes
:*
dtype0
{
block4_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv3/bias
t
%block4_conv3/bias/Read/ReadVariableOpReadVariableOpblock4_conv3/bias*
_output_shapes	
:*
dtype0

block4_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv4/kernel

'block4_conv4/kernel/Read/ReadVariableOpReadVariableOpblock4_conv4/kernel*(
_output_shapes
:*
dtype0
{
block4_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv4/bias
t
%block4_conv4/bias/Read/ReadVariableOpReadVariableOpblock4_conv4/bias*
_output_shapes	
:*
dtype0

block5_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock5_conv1/kernel

'block5_conv1/kernel/Read/ReadVariableOpReadVariableOpblock5_conv1/kernel*(
_output_shapes
:*
dtype0
{
block5_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock5_conv1/bias
t
%block5_conv1/bias/Read/ReadVariableOpReadVariableOpblock5_conv1/bias*
_output_shapes	
:*
dtype0

block5_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock5_conv2/kernel

'block5_conv2/kernel/Read/ReadVariableOpReadVariableOpblock5_conv2/kernel*(
_output_shapes
:*
dtype0
{
block5_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock5_conv2/bias
t
%block5_conv2/bias/Read/ReadVariableOpReadVariableOpblock5_conv2/bias*
_output_shapes	
:*
dtype0

block5_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock5_conv3/kernel

'block5_conv3/kernel/Read/ReadVariableOpReadVariableOpblock5_conv3/kernel*(
_output_shapes
:*
dtype0
{
block5_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock5_conv3/bias
t
%block5_conv3/bias/Read/ReadVariableOpReadVariableOpblock5_conv3/bias*
_output_shapes	
:*
dtype0

block5_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock5_conv4/kernel

'block5_conv4/kernel/Read/ReadVariableOpReadVariableOpblock5_conv4/kernel*(
_output_shapes
:*
dtype0
{
block5_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock5_conv4/bias
t
%block5_conv4/bias/Read/ReadVariableOpReadVariableOpblock5_conv4/bias*
_output_shapes	
:*
dtype0
{
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@* 
shared_namedense_13/kernel
t
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes
:	@*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:@*
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

Adam/dense_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_14/kernel/m

*Adam/dense_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_14/bias/m
y
(Adam/dense_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/m*
_output_shapes
:*
dtype0

Adam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*'
shared_nameAdam/dense_13/kernel/m

*Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/m*
_output_shapes
:	@*
dtype0

Adam/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_13/bias/m
y
(Adam/dense_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_14/kernel/v

*Adam/dense_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_14/bias/v
y
(Adam/dense_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/v*
_output_shapes
:*
dtype0

Adam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*'
shared_nameAdam/dense_13/kernel/v

*Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/v*
_output_shapes
:	@*
dtype0

Adam/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_13/bias/v
y
(Adam/dense_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/v*
_output_shapes
:@*
dtype0

NoOpNoOp
§¯
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*á®
valueÖ®BÒ® BÊ®
Á
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
	optimizer
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
* 
ø
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
¦

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses*

'iter

(beta_1

)beta_2
	*decaym mKmLmv vKvLv*

+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15
;16
<17
=18
>19
?20
@21
A22
B23
C24
D25
E26
F27
G28
H29
I30
J31
K32
L33
34
 35*
 
K0
L1
2
 3*
* 
°
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Rserving_default* 
* 
à
Slayer-0
Tlayer_with_weights-0
Tlayer-1
Ulayer_with_weights-1
Ulayer-2
Vlayer-3
Wlayer_with_weights-2
Wlayer-4
Xlayer_with_weights-3
Xlayer-5
Ylayer-6
Zlayer_with_weights-4
Zlayer-7
[layer_with_weights-5
[layer-8
\layer_with_weights-6
\layer-9
]layer_with_weights-7
]layer-10
^layer-11
_layer_with_weights-8
_layer-12
`layer_with_weights-9
`layer-13
alayer_with_weights-10
alayer-14
blayer_with_weights-11
blayer-15
clayer-16
dlayer_with_weights-12
dlayer-17
elayer_with_weights-13
elayer-18
flayer_with_weights-14
flayer-19
glayer_with_weights-15
glayer-20
hlayer-21
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses*

o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses* 
¦

Kkernel
Lbias
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses*

+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15
;16
<17
=18
>19
?20
@21
A22
B23
C24
D25
E26
F27
G28
H29
I30
J31
K32
L33*

K0
L1*
* 

{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEdense_14/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_14/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
 1*

0
 1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
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
SM
VARIABLE_VALUEblock1_conv1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock1_conv1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock1_conv2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock1_conv2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock2_conv1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock2_conv1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock2_conv2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock2_conv2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock3_conv1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock3_conv1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock3_conv2/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock3_conv2/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock3_conv3/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock3_conv3/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock3_conv4/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock3_conv4/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock4_conv1/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock4_conv1/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock4_conv2/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock4_conv2/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock4_conv3/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock4_conv3/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock4_conv4/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock4_conv4/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock5_conv1/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock5_conv1/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock5_conv2/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock5_conv2/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock5_conv3/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock5_conv3/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock5_conv4/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock5_conv4/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_13/kernel'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_13/bias'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
ú
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15
;16
<17
=18
>19
?20
@21
A22
B23
C24
D25
E26
F27
G28
H29
I30
J31*
'
0
1
2
3
4*

0
1*
* 
* 
* 
* 
¬

+kernel
,bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
¬

-kernel
.bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
¬

/kernel
0bias
	variables
trainable_variables
 regularization_losses
¡	keras_api
¢__call__
+£&call_and_return_all_conditional_losses*
¬

1kernel
2bias
¤	variables
¥trainable_variables
¦regularization_losses
§	keras_api
¨__call__
+©&call_and_return_all_conditional_losses*

ª	variables
«trainable_variables
¬regularization_losses
­	keras_api
®__call__
+¯&call_and_return_all_conditional_losses* 
¬

3kernel
4bias
°	variables
±trainable_variables
²regularization_losses
³	keras_api
´__call__
+µ&call_and_return_all_conditional_losses*
¬

5kernel
6bias
¶	variables
·trainable_variables
¸regularization_losses
¹	keras_api
º__call__
+»&call_and_return_all_conditional_losses*
¬

7kernel
8bias
¼	variables
½trainable_variables
¾regularization_losses
¿	keras_api
À__call__
+Á&call_and_return_all_conditional_losses*
¬

9kernel
:bias
Â	variables
Ãtrainable_variables
Äregularization_losses
Å	keras_api
Æ__call__
+Ç&call_and_return_all_conditional_losses*

È	variables
Étrainable_variables
Êregularization_losses
Ë	keras_api
Ì__call__
+Í&call_and_return_all_conditional_losses* 
¬

;kernel
<bias
Î	variables
Ïtrainable_variables
Ðregularization_losses
Ñ	keras_api
Ò__call__
+Ó&call_and_return_all_conditional_losses*
¬

=kernel
>bias
Ô	variables
Õtrainable_variables
Öregularization_losses
×	keras_api
Ø__call__
+Ù&call_and_return_all_conditional_losses*
¬

?kernel
@bias
Ú	variables
Ûtrainable_variables
Üregularization_losses
Ý	keras_api
Þ__call__
+ß&call_and_return_all_conditional_losses*
¬

Akernel
Bbias
à	variables
átrainable_variables
âregularization_losses
ã	keras_api
ä__call__
+å&call_and_return_all_conditional_losses*

æ	variables
çtrainable_variables
èregularization_losses
é	keras_api
ê__call__
+ë&call_and_return_all_conditional_losses* 
¬

Ckernel
Dbias
ì	variables
ítrainable_variables
îregularization_losses
ï	keras_api
ð__call__
+ñ&call_and_return_all_conditional_losses*
¬

Ekernel
Fbias
ò	variables
ótrainable_variables
ôregularization_losses
õ	keras_api
ö__call__
+÷&call_and_return_all_conditional_losses*
¬

Gkernel
Hbias
ø	variables
ùtrainable_variables
úregularization_losses
û	keras_api
ü__call__
+ý&call_and_return_all_conditional_losses*
¬

Ikernel
Jbias
þ	variables
ÿtrainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
ú
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15
;16
<17
=18
>19
?20
@21
A22
B23
C24
D25
E26
F27
G28
H29
I30
J31*
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses* 
* 
* 

K0
L1*

K0
L1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses*
* 
* 
ú
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15
;16
<17
=18
>19
?20
@21
A22
B23
C24
D25
E26
F27
G28
H29
I30
J31*
 
0
1
2
3*
* 
* 
* 
* 
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

total

count
	variables
	keras_api*
M

total

count

_fn_kwargs
 	variables
¡	keras_api*

+0
,1*
* 
* 

¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 

-0
.1*
* 
* 

§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 

/0
01*
* 
* 

±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
	variables
trainable_variables
 regularization_losses
¢__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses*
* 
* 

10
21*
* 
* 

¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
¤	variables
¥trainable_variables
¦regularization_losses
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

»non_trainable_variables
¼layers
½metrics
 ¾layer_regularization_losses
¿layer_metrics
ª	variables
«trainable_variables
¬regularization_losses
®__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses* 
* 
* 

30
41*
* 
* 

Ànon_trainable_variables
Álayers
Âmetrics
 Ãlayer_regularization_losses
Älayer_metrics
°	variables
±trainable_variables
²regularization_losses
´__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses*
* 
* 

50
61*
* 
* 

Ånon_trainable_variables
Ælayers
Çmetrics
 Èlayer_regularization_losses
Élayer_metrics
¶	variables
·trainable_variables
¸regularization_losses
º__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses*
* 
* 

70
81*
* 
* 

Ênon_trainable_variables
Ëlayers
Ìmetrics
 Ílayer_regularization_losses
Îlayer_metrics
¼	variables
½trainable_variables
¾regularization_losses
À__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses*
* 
* 

90
:1*
* 
* 

Ïnon_trainable_variables
Ðlayers
Ñmetrics
 Òlayer_regularization_losses
Ólayer_metrics
Â	variables
Ãtrainable_variables
Äregularization_losses
Æ__call__
+Ç&call_and_return_all_conditional_losses
'Ç"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ônon_trainable_variables
Õlayers
Ömetrics
 ×layer_regularization_losses
Ølayer_metrics
È	variables
Étrainable_variables
Êregularization_losses
Ì__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses* 
* 
* 

;0
<1*
* 
* 

Ùnon_trainable_variables
Úlayers
Ûmetrics
 Ülayer_regularization_losses
Ýlayer_metrics
Î	variables
Ïtrainable_variables
Ðregularization_losses
Ò__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses*
* 
* 

=0
>1*
* 
* 

Þnon_trainable_variables
ßlayers
àmetrics
 álayer_regularization_losses
âlayer_metrics
Ô	variables
Õtrainable_variables
Öregularization_losses
Ø__call__
+Ù&call_and_return_all_conditional_losses
'Ù"call_and_return_conditional_losses*
* 
* 

?0
@1*
* 
* 

ãnon_trainable_variables
älayers
åmetrics
 ælayer_regularization_losses
çlayer_metrics
Ú	variables
Ûtrainable_variables
Üregularization_losses
Þ__call__
+ß&call_and_return_all_conditional_losses
'ß"call_and_return_conditional_losses*
* 
* 

A0
B1*
* 
* 

ènon_trainable_variables
élayers
êmetrics
 ëlayer_regularization_losses
ìlayer_metrics
à	variables
átrainable_variables
âregularization_losses
ä__call__
+å&call_and_return_all_conditional_losses
'å"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ínon_trainable_variables
îlayers
ïmetrics
 ðlayer_regularization_losses
ñlayer_metrics
æ	variables
çtrainable_variables
èregularization_losses
ê__call__
+ë&call_and_return_all_conditional_losses
'ë"call_and_return_conditional_losses* 
* 
* 

C0
D1*
* 
* 

ònon_trainable_variables
ólayers
ômetrics
 õlayer_regularization_losses
ölayer_metrics
ì	variables
ítrainable_variables
îregularization_losses
ð__call__
+ñ&call_and_return_all_conditional_losses
'ñ"call_and_return_conditional_losses*
* 
* 

E0
F1*
* 
* 

÷non_trainable_variables
ølayers
ùmetrics
 úlayer_regularization_losses
ûlayer_metrics
ò	variables
ótrainable_variables
ôregularization_losses
ö__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses*
* 
* 

G0
H1*
* 
* 

ünon_trainable_variables
ýlayers
þmetrics
 ÿlayer_regularization_losses
layer_metrics
ø	variables
ùtrainable_variables
úregularization_losses
ü__call__
+ý&call_and_return_all_conditional_losses
'ý"call_and_return_conditional_losses*
* 
* 

I0
J1*
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
þ	variables
ÿtrainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
ú
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15
;16
<17
=18
>19
?20
@21
A22
B23
C24
D25
E26
F27
G28
H29
I30
J31*
ª
S0
T1
U2
V3
W4
X5
Y6
Z7
[8
\9
]10
^11
_12
`13
a14
b15
c16
d17
e18
f19
g20
h21*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

 	variables*

+0
,1*
* 
* 
* 
* 

-0
.1*
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
/0
01*
* 
* 
* 
* 

10
21*
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
30
41*
* 
* 
* 
* 

50
61*
* 
* 
* 
* 

70
81*
* 
* 
* 
* 

90
:1*
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
;0
<1*
* 
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

?0
@1*
* 
* 
* 
* 

A0
B1*
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
C0
D1*
* 
* 
* 
* 

E0
F1*
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

I0
J1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
|
VARIABLE_VALUEAdam/dense_14/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_14/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_13/kernel/mCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_13/bias/mCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_14/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_14/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/dense_13/kernel/vCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/dense_13/bias/vCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_27Placeholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿÈÈ

serving_default_input_28Placeholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿÈÈ
Ù
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_27serving_default_input_28block1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock3_conv4/kernelblock3_conv4/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock4_conv4/kernelblock4_conv4/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biasblock5_conv4/kernelblock5_conv4/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/bias*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_977982
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
«
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp'block1_conv1/kernel/Read/ReadVariableOp%block1_conv1/bias/Read/ReadVariableOp'block1_conv2/kernel/Read/ReadVariableOp%block1_conv2/bias/Read/ReadVariableOp'block2_conv1/kernel/Read/ReadVariableOp%block2_conv1/bias/Read/ReadVariableOp'block2_conv2/kernel/Read/ReadVariableOp%block2_conv2/bias/Read/ReadVariableOp'block3_conv1/kernel/Read/ReadVariableOp%block3_conv1/bias/Read/ReadVariableOp'block3_conv2/kernel/Read/ReadVariableOp%block3_conv2/bias/Read/ReadVariableOp'block3_conv3/kernel/Read/ReadVariableOp%block3_conv3/bias/Read/ReadVariableOp'block3_conv4/kernel/Read/ReadVariableOp%block3_conv4/bias/Read/ReadVariableOp'block4_conv1/kernel/Read/ReadVariableOp%block4_conv1/bias/Read/ReadVariableOp'block4_conv2/kernel/Read/ReadVariableOp%block4_conv2/bias/Read/ReadVariableOp'block4_conv3/kernel/Read/ReadVariableOp%block4_conv3/bias/Read/ReadVariableOp'block4_conv4/kernel/Read/ReadVariableOp%block4_conv4/bias/Read/ReadVariableOp'block5_conv1/kernel/Read/ReadVariableOp%block5_conv1/bias/Read/ReadVariableOp'block5_conv2/kernel/Read/ReadVariableOp%block5_conv2/bias/Read/ReadVariableOp'block5_conv3/kernel/Read/ReadVariableOp%block5_conv3/bias/Read/ReadVariableOp'block5_conv4/kernel/Read/ReadVariableOp%block5_conv4/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_14/kernel/m/Read/ReadVariableOp(Adam/dense_14/bias/m/Read/ReadVariableOp*Adam/dense_13/kernel/m/Read/ReadVariableOp(Adam/dense_13/bias/m/Read/ReadVariableOp*Adam/dense_14/kernel/v/Read/ReadVariableOp(Adam/dense_14/bias/v/Read/ReadVariableOp*Adam/dense_13/kernel/v/Read/ReadVariableOp(Adam/dense_13/bias/v/Read/ReadVariableOpConst*A
Tin:
826	*
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
GPU2*0J 8 *(
f#R!
__inference__traced_save_979406


StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_14/kerneldense_14/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock3_conv4/kernelblock3_conv4/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock4_conv4/kernelblock4_conv4/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biasblock5_conv4/kernelblock5_conv4/biasdense_13/kerneldense_13/biastotalcounttotal_1count_1Adam/dense_14/kernel/mAdam/dense_14/bias/mAdam/dense_13/kernel/mAdam/dense_13/bias/mAdam/dense_14/kernel/vAdam/dense_14/bias/vAdam/dense_13/kernel/vAdam/dense_13/bias/v*@
Tin9
725*
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
GPU2*0J 8 *+
f&R$
"__inference__traced_restore_979572ß­


H__inference_block4_conv4_layer_call_and_return_conditional_losses_979126

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block4_conv1_layer_call_and_return_conditional_losses_974973

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block4_conv2_layer_call_and_return_conditional_losses_979086

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë	
ö
D__inference_dense_13_layer_call_and_return_conditional_losses_975898

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block4_conv2_layer_call_and_return_conditional_losses_974990

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


p
D__inference_lambda_4_layer_call_and_return_conditional_losses_978412
inputs_0
inputs_1
identityP
subSubinputs_0inputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@K
SquareSquaresub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :y
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3f
MaximumMaximumSum:output:0Maximum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    c
	Maximum_1MaximumMaximum:z:0Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
SqrtSqrtMaximum_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/1


õ
D__inference_dense_14_layer_call_and_return_conditional_losses_978446

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

c
G__inference_block3_pool_layer_call_and_return_conditional_losses_974789

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
ÿ	
)__inference_model_14_layer_call_fn_977428
inputs_0
inputs_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	

unknown_31:	@

unknown_32:@

unknown_33:

unknown_34:
identity¢StatefulPartitionedCall¼
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
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_14_layer_call_and_return_conditional_losses_976885o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
inputs/1
ª
ý
&__inference_vgg19_layer_call_fn_975624
input_30!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinput_30unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_vgg19_layer_call_and_return_conditional_losses_975488x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
input_30


n
D__inference_lambda_4_layer_call_and_return_conditional_losses_976686

inputs
inputs_1
identityN
subSubinputsinputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@K
SquareSquaresub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :y
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3f
MaximumMaximumSum:output:0Maximum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    c
	Maximum_1MaximumMaximum:z:0Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
SqrtSqrtMaximum_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ý
¢
-__inference_block1_conv1_layer_call_fn_978865

inputs!
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block1_conv1_layer_call_and_return_conditional_losses_974834y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÈÈ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs
ü
¥
-__inference_block4_conv4_layer_call_fn_979115

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv4_layer_call_and_return_conditional_losses_975024x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block1_conv1_layer_call_and_return_conditional_losses_978876

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÈÈ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs

c
G__inference_block2_pool_layer_call_and_return_conditional_losses_978956

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

c
G__inference_block3_pool_layer_call_and_return_conditional_losses_979046

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block2_conv2_layer_call_and_return_conditional_losses_978946

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿdd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
³
H
,__inference_block2_pool_layer_call_fn_978951

inputs
identityØ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block2_pool_layer_call_and_return_conditional_losses_974777
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ãÆ
¾ 
"__inference__traced_restore_979572
file_prefix2
 assignvariableop_dense_14_kernel:.
 assignvariableop_1_dense_14_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: @
&assignvariableop_6_block1_conv1_kernel:@2
$assignvariableop_7_block1_conv1_bias:@@
&assignvariableop_8_block1_conv2_kernel:@@2
$assignvariableop_9_block1_conv2_bias:@B
'assignvariableop_10_block2_conv1_kernel:@4
%assignvariableop_11_block2_conv1_bias:	C
'assignvariableop_12_block2_conv2_kernel:4
%assignvariableop_13_block2_conv2_bias:	C
'assignvariableop_14_block3_conv1_kernel:4
%assignvariableop_15_block3_conv1_bias:	C
'assignvariableop_16_block3_conv2_kernel:4
%assignvariableop_17_block3_conv2_bias:	C
'assignvariableop_18_block3_conv3_kernel:4
%assignvariableop_19_block3_conv3_bias:	C
'assignvariableop_20_block3_conv4_kernel:4
%assignvariableop_21_block3_conv4_bias:	C
'assignvariableop_22_block4_conv1_kernel:4
%assignvariableop_23_block4_conv1_bias:	C
'assignvariableop_24_block4_conv2_kernel:4
%assignvariableop_25_block4_conv2_bias:	C
'assignvariableop_26_block4_conv3_kernel:4
%assignvariableop_27_block4_conv3_bias:	C
'assignvariableop_28_block4_conv4_kernel:4
%assignvariableop_29_block4_conv4_bias:	C
'assignvariableop_30_block5_conv1_kernel:4
%assignvariableop_31_block5_conv1_bias:	C
'assignvariableop_32_block5_conv2_kernel:4
%assignvariableop_33_block5_conv2_bias:	C
'assignvariableop_34_block5_conv3_kernel:4
%assignvariableop_35_block5_conv3_bias:	C
'assignvariableop_36_block5_conv4_kernel:4
%assignvariableop_37_block5_conv4_bias:	6
#assignvariableop_38_dense_13_kernel:	@/
!assignvariableop_39_dense_13_bias:@#
assignvariableop_40_total: #
assignvariableop_41_count: %
assignvariableop_42_total_1: %
assignvariableop_43_count_1: <
*assignvariableop_44_adam_dense_14_kernel_m:6
(assignvariableop_45_adam_dense_14_bias_m:=
*assignvariableop_46_adam_dense_13_kernel_m:	@6
(assignvariableop_47_adam_dense_13_bias_m:@<
*assignvariableop_48_adam_dense_14_kernel_v:6
(assignvariableop_49_adam_dense_14_bias_v:=
*assignvariableop_50_adam_dense_13_kernel_v:	@6
(assignvariableop_51_adam_dense_13_bias_v:@
identity_53¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ì
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*ò
valueèBå5B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÚ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ª
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ê
_output_shapes×
Ô:::::::::::::::::::::::::::::::::::::::::::::::::::::*C
dtypes9
725	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp assignvariableop_dense_14_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_14_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp&assignvariableop_6_block1_conv1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp$assignvariableop_7_block1_conv1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp&assignvariableop_8_block1_conv2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp$assignvariableop_9_block1_conv2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp'assignvariableop_10_block2_conv1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp%assignvariableop_11_block2_conv1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp'assignvariableop_12_block2_conv2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp%assignvariableop_13_block2_conv2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp'assignvariableop_14_block3_conv1_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp%assignvariableop_15_block3_conv1_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp'assignvariableop_16_block3_conv2_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp%assignvariableop_17_block3_conv2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp'assignvariableop_18_block3_conv3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp%assignvariableop_19_block3_conv3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp'assignvariableop_20_block3_conv4_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp%assignvariableop_21_block3_conv4_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp'assignvariableop_22_block4_conv1_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp%assignvariableop_23_block4_conv1_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp'assignvariableop_24_block4_conv2_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp%assignvariableop_25_block4_conv2_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp'assignvariableop_26_block4_conv3_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp%assignvariableop_27_block4_conv3_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp'assignvariableop_28_block4_conv4_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp%assignvariableop_29_block4_conv4_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp'assignvariableop_30_block5_conv1_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp%assignvariableop_31_block5_conv1_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp'assignvariableop_32_block5_conv2_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp%assignvariableop_33_block5_conv2_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp'assignvariableop_34_block5_conv3_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp%assignvariableop_35_block5_conv3_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp'assignvariableop_36_block5_conv4_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp%assignvariableop_37_block5_conv4_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp#assignvariableop_38_dense_13_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp!assignvariableop_39_dense_13_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOpassignvariableop_40_totalIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOpassignvariableop_41_countIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOpassignvariableop_42_total_1Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOpassignvariableop_43_count_1Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_dense_14_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_dense_14_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp*assignvariableop_46_adam_dense_13_kernel_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_dense_13_bias_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_dense_14_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_dense_14_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOp*assignvariableop_50_adam_dense_13_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_dense_13_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ç	
Identity_52Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_53IdentityIdentity_52:output:0^NoOp_1*
T0*
_output_shapes
: ´	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_53Identity_53:output:0*}
_input_shapesl
j: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_51AssignVariableOp_512(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ô
Ò

D__inference_model_13_layer_call_and_return_conditional_losses_976136

inputs&
vgg19_976064:@
vgg19_976066:@&
vgg19_976068:@@
vgg19_976070:@'
vgg19_976072:@
vgg19_976074:	(
vgg19_976076:
vgg19_976078:	(
vgg19_976080:
vgg19_976082:	(
vgg19_976084:
vgg19_976086:	(
vgg19_976088:
vgg19_976090:	(
vgg19_976092:
vgg19_976094:	(
vgg19_976096:
vgg19_976098:	(
vgg19_976100:
vgg19_976102:	(
vgg19_976104:
vgg19_976106:	(
vgg19_976108:
vgg19_976110:	(
vgg19_976112:
vgg19_976114:	(
vgg19_976116:
vgg19_976118:	(
vgg19_976120:
vgg19_976122:	(
vgg19_976124:
vgg19_976126:	"
dense_13_976130:	@
dense_13_976132:@
identity¢ dense_13/StatefulPartitionedCall¢vgg19/StatefulPartitionedCallÐ
vgg19/StatefulPartitionedCallStatefulPartitionedCallinputsvgg19_976064vgg19_976066vgg19_976068vgg19_976070vgg19_976072vgg19_976074vgg19_976076vgg19_976078vgg19_976080vgg19_976082vgg19_976084vgg19_976086vgg19_976088vgg19_976090vgg19_976092vgg19_976094vgg19_976096vgg19_976098vgg19_976100vgg19_976102vgg19_976104vgg19_976106vgg19_976108vgg19_976110vgg19_976112vgg19_976114vgg19_976116vgg19_976118vgg19_976120vgg19_976122vgg19_976124vgg19_976126*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_vgg19_layer_call_and_return_conditional_losses_975488
*global_average_pooling2d_9/PartitionedCallPartitionedCall&vgg19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *_
fZRX
V__inference_global_average_pooling2d_9_layer_call_and_return_conditional_losses_975812 
 dense_13/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_9/PartitionedCall:output:0dense_13_976130dense_13_976132*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_975898x
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp!^dense_13/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2>
vgg19/StatefulPartitionedCallvgg19/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs
å
ÿ	
)__inference_model_14_layer_call_fn_977350
inputs_0
inputs_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	

unknown_31:	@

unknown_32:@

unknown_33:

unknown_34:
identity¢StatefulPartitionedCall¼
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
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_14_layer_call_and_return_conditional_losses_976577o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
inputs/1
ü
¥
-__inference_block5_conv3_layer_call_fn_979185

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv3_layer_call_and_return_conditional_losses_975076x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block3_conv4_layer_call_and_return_conditional_losses_974955

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
ü
¥
-__inference_block3_conv2_layer_call_fn_978985

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv2_layer_call_and_return_conditional_losses_974921x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ22: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
ñ
·	
)__inference_model_13_layer_call_fn_978128

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	

unknown_31:	@

unknown_32:@
identity¢StatefulPartitionedCall
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
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_13_layer_call_and_return_conditional_losses_976136o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs
¿
£
D__inference_model_13_layer_call_and_return_conditional_losses_978257

inputsK
1vgg19_block1_conv1_conv2d_readvariableop_resource:@@
2vgg19_block1_conv1_biasadd_readvariableop_resource:@K
1vgg19_block1_conv2_conv2d_readvariableop_resource:@@@
2vgg19_block1_conv2_biasadd_readvariableop_resource:@L
1vgg19_block2_conv1_conv2d_readvariableop_resource:@A
2vgg19_block2_conv1_biasadd_readvariableop_resource:	M
1vgg19_block2_conv2_conv2d_readvariableop_resource:A
2vgg19_block2_conv2_biasadd_readvariableop_resource:	M
1vgg19_block3_conv1_conv2d_readvariableop_resource:A
2vgg19_block3_conv1_biasadd_readvariableop_resource:	M
1vgg19_block3_conv2_conv2d_readvariableop_resource:A
2vgg19_block3_conv2_biasadd_readvariableop_resource:	M
1vgg19_block3_conv3_conv2d_readvariableop_resource:A
2vgg19_block3_conv3_biasadd_readvariableop_resource:	M
1vgg19_block3_conv4_conv2d_readvariableop_resource:A
2vgg19_block3_conv4_biasadd_readvariableop_resource:	M
1vgg19_block4_conv1_conv2d_readvariableop_resource:A
2vgg19_block4_conv1_biasadd_readvariableop_resource:	M
1vgg19_block4_conv2_conv2d_readvariableop_resource:A
2vgg19_block4_conv2_biasadd_readvariableop_resource:	M
1vgg19_block4_conv3_conv2d_readvariableop_resource:A
2vgg19_block4_conv3_biasadd_readvariableop_resource:	M
1vgg19_block4_conv4_conv2d_readvariableop_resource:A
2vgg19_block4_conv4_biasadd_readvariableop_resource:	M
1vgg19_block5_conv1_conv2d_readvariableop_resource:A
2vgg19_block5_conv1_biasadd_readvariableop_resource:	M
1vgg19_block5_conv2_conv2d_readvariableop_resource:A
2vgg19_block5_conv2_biasadd_readvariableop_resource:	M
1vgg19_block5_conv3_conv2d_readvariableop_resource:A
2vgg19_block5_conv3_biasadd_readvariableop_resource:	M
1vgg19_block5_conv4_conv2d_readvariableop_resource:A
2vgg19_block5_conv4_biasadd_readvariableop_resource:	:
'dense_13_matmul_readvariableop_resource:	@6
(dense_13_biasadd_readvariableop_resource:@
identity¢dense_13/BiasAdd/ReadVariableOp¢dense_13/MatMul/ReadVariableOp¢)vgg19/block1_conv1/BiasAdd/ReadVariableOp¢(vgg19/block1_conv1/Conv2D/ReadVariableOp¢)vgg19/block1_conv2/BiasAdd/ReadVariableOp¢(vgg19/block1_conv2/Conv2D/ReadVariableOp¢)vgg19/block2_conv1/BiasAdd/ReadVariableOp¢(vgg19/block2_conv1/Conv2D/ReadVariableOp¢)vgg19/block2_conv2/BiasAdd/ReadVariableOp¢(vgg19/block2_conv2/Conv2D/ReadVariableOp¢)vgg19/block3_conv1/BiasAdd/ReadVariableOp¢(vgg19/block3_conv1/Conv2D/ReadVariableOp¢)vgg19/block3_conv2/BiasAdd/ReadVariableOp¢(vgg19/block3_conv2/Conv2D/ReadVariableOp¢)vgg19/block3_conv3/BiasAdd/ReadVariableOp¢(vgg19/block3_conv3/Conv2D/ReadVariableOp¢)vgg19/block3_conv4/BiasAdd/ReadVariableOp¢(vgg19/block3_conv4/Conv2D/ReadVariableOp¢)vgg19/block4_conv1/BiasAdd/ReadVariableOp¢(vgg19/block4_conv1/Conv2D/ReadVariableOp¢)vgg19/block4_conv2/BiasAdd/ReadVariableOp¢(vgg19/block4_conv2/Conv2D/ReadVariableOp¢)vgg19/block4_conv3/BiasAdd/ReadVariableOp¢(vgg19/block4_conv3/Conv2D/ReadVariableOp¢)vgg19/block4_conv4/BiasAdd/ReadVariableOp¢(vgg19/block4_conv4/Conv2D/ReadVariableOp¢)vgg19/block5_conv1/BiasAdd/ReadVariableOp¢(vgg19/block5_conv1/Conv2D/ReadVariableOp¢)vgg19/block5_conv2/BiasAdd/ReadVariableOp¢(vgg19/block5_conv2/Conv2D/ReadVariableOp¢)vgg19/block5_conv3/BiasAdd/ReadVariableOp¢(vgg19/block5_conv3/Conv2D/ReadVariableOp¢)vgg19/block5_conv4/BiasAdd/ReadVariableOp¢(vgg19/block5_conv4/Conv2D/ReadVariableOp¢
(vgg19/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Á
vgg19/block1_conv1/Conv2DConv2Dinputs0vgg19/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides

)vgg19/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¸
vgg19/block1_conv1/BiasAddBiasAdd"vgg19/block1_conv1/Conv2D:output:01vgg19/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
vgg19/block1_conv1/ReluRelu#vgg19/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@¢
(vgg19/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0à
vgg19/block1_conv2/Conv2DConv2D%vgg19/block1_conv1/Relu:activations:00vgg19/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides

)vgg19/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¸
vgg19/block1_conv2/BiasAddBiasAdd"vgg19/block1_conv2/Conv2D:output:01vgg19/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
vgg19/block1_conv2/ReluRelu#vgg19/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@¸
vgg19/block1_pool/MaxPoolMaxPool%vgg19/block1_conv2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@*
ksize
*
paddingVALID*
strides
£
(vgg19/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ü
vgg19/block2_conv1/Conv2DConv2D"vgg19/block1_pool/MaxPool:output:00vgg19/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides

)vgg19/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block2_conv1/BiasAddBiasAdd"vgg19/block2_conv1/Conv2D:output:01vgg19/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
vgg19/block2_conv1/ReluRelu#vgg19/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd¤
(vgg19/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block2_conv2/Conv2DConv2D%vgg19/block2_conv1/Relu:activations:00vgg19/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides

)vgg19/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block2_conv2/BiasAddBiasAdd"vgg19/block2_conv2/Conv2D:output:01vgg19/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
vgg19/block2_conv2/ReluRelu#vgg19/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd¹
vgg19/block2_pool/MaxPoolMaxPool%vgg19/block2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
ksize
*
paddingVALID*
strides
¤
(vgg19/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
vgg19/block3_conv1/Conv2DConv2D"vgg19/block2_pool/MaxPool:output:00vgg19/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

)vgg19/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block3_conv1/BiasAddBiasAdd"vgg19/block3_conv1/Conv2D:output:01vgg19/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
vgg19/block3_conv1/ReluRelu#vgg19/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¤
(vgg19/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block3_conv2/Conv2DConv2D%vgg19/block3_conv1/Relu:activations:00vgg19/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

)vgg19/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block3_conv2/BiasAddBiasAdd"vgg19/block3_conv2/Conv2D:output:01vgg19/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
vgg19/block3_conv2/ReluRelu#vgg19/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¤
(vgg19/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block3_conv3/Conv2DConv2D%vgg19/block3_conv2/Relu:activations:00vgg19/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

)vgg19/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block3_conv3/BiasAddBiasAdd"vgg19/block3_conv3/Conv2D:output:01vgg19/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
vgg19/block3_conv3/ReluRelu#vgg19/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¤
(vgg19/block3_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block3_conv4/Conv2DConv2D%vgg19/block3_conv3/Relu:activations:00vgg19/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

)vgg19/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block3_conv4/BiasAddBiasAdd"vgg19/block3_conv4/Conv2D:output:01vgg19/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
vgg19/block3_conv4/ReluRelu#vgg19/block3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¹
vgg19/block3_pool/MaxPoolMaxPool%vgg19/block3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¤
(vgg19/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
vgg19/block4_conv1/Conv2DConv2D"vgg19/block3_pool/MaxPool:output:00vgg19/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block4_conv1/BiasAddBiasAdd"vgg19/block4_conv1/Conv2D:output:01vgg19/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block4_conv1/ReluRelu#vgg19/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg19/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block4_conv2/Conv2DConv2D%vgg19/block4_conv1/Relu:activations:00vgg19/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block4_conv2/BiasAddBiasAdd"vgg19/block4_conv2/Conv2D:output:01vgg19/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block4_conv2/ReluRelu#vgg19/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg19/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block4_conv3/Conv2DConv2D%vgg19/block4_conv2/Relu:activations:00vgg19/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block4_conv3/BiasAddBiasAdd"vgg19/block4_conv3/Conv2D:output:01vgg19/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block4_conv3/ReluRelu#vgg19/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg19/block4_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block4_conv4/Conv2DConv2D%vgg19/block4_conv3/Relu:activations:00vgg19/block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block4_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block4_conv4/BiasAddBiasAdd"vgg19/block4_conv4/Conv2D:output:01vgg19/block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block4_conv4/ReluRelu#vgg19/block4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
vgg19/block4_pool/MaxPoolMaxPool%vgg19/block4_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¤
(vgg19/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
vgg19/block5_conv1/Conv2DConv2D"vgg19/block4_pool/MaxPool:output:00vgg19/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block5_conv1/BiasAddBiasAdd"vgg19/block5_conv1/Conv2D:output:01vgg19/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block5_conv1/ReluRelu#vgg19/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg19/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block5_conv2/Conv2DConv2D%vgg19/block5_conv1/Relu:activations:00vgg19/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block5_conv2/BiasAddBiasAdd"vgg19/block5_conv2/Conv2D:output:01vgg19/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block5_conv2/ReluRelu#vgg19/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg19/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block5_conv3/Conv2DConv2D%vgg19/block5_conv2/Relu:activations:00vgg19/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block5_conv3/BiasAddBiasAdd"vgg19/block5_conv3/Conv2D:output:01vgg19/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block5_conv3/ReluRelu#vgg19/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg19/block5_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block5_conv4/Conv2DConv2D%vgg19/block5_conv3/Relu:activations:00vgg19/block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block5_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block5_conv4/BiasAddBiasAdd"vgg19/block5_conv4/Conv2D:output:01vgg19/block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block5_conv4/ReluRelu#vgg19/block5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
vgg19/block5_pool/MaxPoolMaxPool%vgg19/block5_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

1global_average_pooling2d_9/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      º
global_average_pooling2d_9/MeanMean"vgg19/block5_pool/MaxPool:output:0:global_average_pooling2d_9/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
dense_13/MatMulMatMul(global_average_pooling2d_9/Mean:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
IdentityIdentitydense_13/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ù
NoOpNoOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*^vgg19/block1_conv1/BiasAdd/ReadVariableOp)^vgg19/block1_conv1/Conv2D/ReadVariableOp*^vgg19/block1_conv2/BiasAdd/ReadVariableOp)^vgg19/block1_conv2/Conv2D/ReadVariableOp*^vgg19/block2_conv1/BiasAdd/ReadVariableOp)^vgg19/block2_conv1/Conv2D/ReadVariableOp*^vgg19/block2_conv2/BiasAdd/ReadVariableOp)^vgg19/block2_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv1/BiasAdd/ReadVariableOp)^vgg19/block3_conv1/Conv2D/ReadVariableOp*^vgg19/block3_conv2/BiasAdd/ReadVariableOp)^vgg19/block3_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv3/BiasAdd/ReadVariableOp)^vgg19/block3_conv3/Conv2D/ReadVariableOp*^vgg19/block3_conv4/BiasAdd/ReadVariableOp)^vgg19/block3_conv4/Conv2D/ReadVariableOp*^vgg19/block4_conv1/BiasAdd/ReadVariableOp)^vgg19/block4_conv1/Conv2D/ReadVariableOp*^vgg19/block4_conv2/BiasAdd/ReadVariableOp)^vgg19/block4_conv2/Conv2D/ReadVariableOp*^vgg19/block4_conv3/BiasAdd/ReadVariableOp)^vgg19/block4_conv3/Conv2D/ReadVariableOp*^vgg19/block4_conv4/BiasAdd/ReadVariableOp)^vgg19/block4_conv4/Conv2D/ReadVariableOp*^vgg19/block5_conv1/BiasAdd/ReadVariableOp)^vgg19/block5_conv1/Conv2D/ReadVariableOp*^vgg19/block5_conv2/BiasAdd/ReadVariableOp)^vgg19/block5_conv2/Conv2D/ReadVariableOp*^vgg19/block5_conv3/BiasAdd/ReadVariableOp)^vgg19/block5_conv3/Conv2D/ReadVariableOp*^vgg19/block5_conv4/BiasAdd/ReadVariableOp)^vgg19/block5_conv4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2V
)vgg19/block1_conv1/BiasAdd/ReadVariableOp)vgg19/block1_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block1_conv1/Conv2D/ReadVariableOp(vgg19/block1_conv1/Conv2D/ReadVariableOp2V
)vgg19/block1_conv2/BiasAdd/ReadVariableOp)vgg19/block1_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block1_conv2/Conv2D/ReadVariableOp(vgg19/block1_conv2/Conv2D/ReadVariableOp2V
)vgg19/block2_conv1/BiasAdd/ReadVariableOp)vgg19/block2_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block2_conv1/Conv2D/ReadVariableOp(vgg19/block2_conv1/Conv2D/ReadVariableOp2V
)vgg19/block2_conv2/BiasAdd/ReadVariableOp)vgg19/block2_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block2_conv2/Conv2D/ReadVariableOp(vgg19/block2_conv2/Conv2D/ReadVariableOp2V
)vgg19/block3_conv1/BiasAdd/ReadVariableOp)vgg19/block3_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv1/Conv2D/ReadVariableOp(vgg19/block3_conv1/Conv2D/ReadVariableOp2V
)vgg19/block3_conv2/BiasAdd/ReadVariableOp)vgg19/block3_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv2/Conv2D/ReadVariableOp(vgg19/block3_conv2/Conv2D/ReadVariableOp2V
)vgg19/block3_conv3/BiasAdd/ReadVariableOp)vgg19/block3_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv3/Conv2D/ReadVariableOp(vgg19/block3_conv3/Conv2D/ReadVariableOp2V
)vgg19/block3_conv4/BiasAdd/ReadVariableOp)vgg19/block3_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv4/Conv2D/ReadVariableOp(vgg19/block3_conv4/Conv2D/ReadVariableOp2V
)vgg19/block4_conv1/BiasAdd/ReadVariableOp)vgg19/block4_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv1/Conv2D/ReadVariableOp(vgg19/block4_conv1/Conv2D/ReadVariableOp2V
)vgg19/block4_conv2/BiasAdd/ReadVariableOp)vgg19/block4_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv2/Conv2D/ReadVariableOp(vgg19/block4_conv2/Conv2D/ReadVariableOp2V
)vgg19/block4_conv3/BiasAdd/ReadVariableOp)vgg19/block4_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv3/Conv2D/ReadVariableOp(vgg19/block4_conv3/Conv2D/ReadVariableOp2V
)vgg19/block4_conv4/BiasAdd/ReadVariableOp)vgg19/block4_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv4/Conv2D/ReadVariableOp(vgg19/block4_conv4/Conv2D/ReadVariableOp2V
)vgg19/block5_conv1/BiasAdd/ReadVariableOp)vgg19/block5_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv1/Conv2D/ReadVariableOp(vgg19/block5_conv1/Conv2D/ReadVariableOp2V
)vgg19/block5_conv2/BiasAdd/ReadVariableOp)vgg19/block5_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv2/Conv2D/ReadVariableOp(vgg19/block5_conv2/Conv2D/ReadVariableOp2V
)vgg19/block5_conv3/BiasAdd/ReadVariableOp)vgg19/block5_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv3/Conv2D/ReadVariableOp(vgg19/block5_conv3/Conv2D/ReadVariableOp2V
)vgg19/block5_conv4/BiasAdd/ReadVariableOp)vgg19/block5_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv4/Conv2D/ReadVariableOp(vgg19/block5_conv4/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs
¢)
¬
D__inference_model_14_layer_call_and_return_conditional_losses_977268
input_27
input_28)
model_13_977157:@
model_13_977159:@)
model_13_977161:@@
model_13_977163:@*
model_13_977165:@
model_13_977167:	+
model_13_977169:
model_13_977171:	+
model_13_977173:
model_13_977175:	+
model_13_977177:
model_13_977179:	+
model_13_977181:
model_13_977183:	+
model_13_977185:
model_13_977187:	+
model_13_977189:
model_13_977191:	+
model_13_977193:
model_13_977195:	+
model_13_977197:
model_13_977199:	+
model_13_977201:
model_13_977203:	+
model_13_977205:
model_13_977207:	+
model_13_977209:
model_13_977211:	+
model_13_977213:
model_13_977215:	+
model_13_977217:
model_13_977219:	"
model_13_977221:	@
model_13_977223:@!
dense_14_977262:
dense_14_977264:
identity¢ dense_14/StatefulPartitionedCall¢ model_13/StatefulPartitionedCall¢"model_13/StatefulPartitionedCall_1Õ
 model_13/StatefulPartitionedCallStatefulPartitionedCallinput_27model_13_977157model_13_977159model_13_977161model_13_977163model_13_977165model_13_977167model_13_977169model_13_977171model_13_977173model_13_977175model_13_977177model_13_977179model_13_977181model_13_977183model_13_977185model_13_977187model_13_977189model_13_977191model_13_977193model_13_977195model_13_977197model_13_977199model_13_977201model_13_977203model_13_977205model_13_977207model_13_977209model_13_977211model_13_977213model_13_977215model_13_977217model_13_977219model_13_977221model_13_977223*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_13_layer_call_and_return_conditional_losses_976136×
"model_13/StatefulPartitionedCall_1StatefulPartitionedCallinput_28model_13_977157model_13_977159model_13_977161model_13_977163model_13_977165model_13_977167model_13_977169model_13_977171model_13_977173model_13_977175model_13_977177model_13_977179model_13_977181model_13_977183model_13_977185model_13_977187model_13_977189model_13_977191model_13_977193model_13_977195model_13_977197model_13_977199model_13_977201model_13_977203model_13_977205model_13_977207model_13_977209model_13_977211model_13_977213model_13_977215model_13_977217model_13_977219model_13_977221model_13_977223*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_13_layer_call_and_return_conditional_losses_976136
lambda_4/PartitionedCallPartitionedCall)model_13/StatefulPartitionedCall:output:0+model_13/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_lambda_4_layer_call_and_return_conditional_losses_976686
 dense_14/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0dense_14_977262dense_14_977264*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_976570x
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
NoOpNoOp!^dense_14/StatefulPartitionedCall!^model_13/StatefulPartitionedCall#^model_13/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 model_13/StatefulPartitionedCall model_13/StatefulPartitionedCall2H
"model_13/StatefulPartitionedCall_1"model_13/StatefulPartitionedCall_1:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
input_27:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
input_28
ú
Ô

D__inference_model_13_layer_call_and_return_conditional_losses_976355
input_29&
vgg19_976283:@
vgg19_976285:@&
vgg19_976287:@@
vgg19_976289:@'
vgg19_976291:@
vgg19_976293:	(
vgg19_976295:
vgg19_976297:	(
vgg19_976299:
vgg19_976301:	(
vgg19_976303:
vgg19_976305:	(
vgg19_976307:
vgg19_976309:	(
vgg19_976311:
vgg19_976313:	(
vgg19_976315:
vgg19_976317:	(
vgg19_976319:
vgg19_976321:	(
vgg19_976323:
vgg19_976325:	(
vgg19_976327:
vgg19_976329:	(
vgg19_976331:
vgg19_976333:	(
vgg19_976335:
vgg19_976337:	(
vgg19_976339:
vgg19_976341:	(
vgg19_976343:
vgg19_976345:	"
dense_13_976349:	@
dense_13_976351:@
identity¢ dense_13/StatefulPartitionedCall¢vgg19/StatefulPartitionedCallÒ
vgg19/StatefulPartitionedCallStatefulPartitionedCallinput_29vgg19_976283vgg19_976285vgg19_976287vgg19_976289vgg19_976291vgg19_976293vgg19_976295vgg19_976297vgg19_976299vgg19_976301vgg19_976303vgg19_976305vgg19_976307vgg19_976309vgg19_976311vgg19_976313vgg19_976315vgg19_976317vgg19_976319vgg19_976321vgg19_976323vgg19_976325vgg19_976327vgg19_976329vgg19_976331vgg19_976333vgg19_976335vgg19_976337vgg19_976339vgg19_976341vgg19_976343vgg19_976345*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_vgg19_layer_call_and_return_conditional_losses_975101
*global_average_pooling2d_9/PartitionedCallPartitionedCall&vgg19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *_
fZRX
V__inference_global_average_pooling2d_9_layer_call_and_return_conditional_losses_975812 
 dense_13/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_9/PartitionedCall:output:0dense_13_976349dense_13_976351*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_975898x
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp!^dense_13/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2>
vgg19/StatefulPartitionedCallvgg19/StatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
input_29
±
ö3
D__inference_model_14_layer_call_and_return_conditional_losses_977665
inputs_0
inputs_1T
:model_13_vgg19_block1_conv1_conv2d_readvariableop_resource:@I
;model_13_vgg19_block1_conv1_biasadd_readvariableop_resource:@T
:model_13_vgg19_block1_conv2_conv2d_readvariableop_resource:@@I
;model_13_vgg19_block1_conv2_biasadd_readvariableop_resource:@U
:model_13_vgg19_block2_conv1_conv2d_readvariableop_resource:@J
;model_13_vgg19_block2_conv1_biasadd_readvariableop_resource:	V
:model_13_vgg19_block2_conv2_conv2d_readvariableop_resource:J
;model_13_vgg19_block2_conv2_biasadd_readvariableop_resource:	V
:model_13_vgg19_block3_conv1_conv2d_readvariableop_resource:J
;model_13_vgg19_block3_conv1_biasadd_readvariableop_resource:	V
:model_13_vgg19_block3_conv2_conv2d_readvariableop_resource:J
;model_13_vgg19_block3_conv2_biasadd_readvariableop_resource:	V
:model_13_vgg19_block3_conv3_conv2d_readvariableop_resource:J
;model_13_vgg19_block3_conv3_biasadd_readvariableop_resource:	V
:model_13_vgg19_block3_conv4_conv2d_readvariableop_resource:J
;model_13_vgg19_block3_conv4_biasadd_readvariableop_resource:	V
:model_13_vgg19_block4_conv1_conv2d_readvariableop_resource:J
;model_13_vgg19_block4_conv1_biasadd_readvariableop_resource:	V
:model_13_vgg19_block4_conv2_conv2d_readvariableop_resource:J
;model_13_vgg19_block4_conv2_biasadd_readvariableop_resource:	V
:model_13_vgg19_block4_conv3_conv2d_readvariableop_resource:J
;model_13_vgg19_block4_conv3_biasadd_readvariableop_resource:	V
:model_13_vgg19_block4_conv4_conv2d_readvariableop_resource:J
;model_13_vgg19_block4_conv4_biasadd_readvariableop_resource:	V
:model_13_vgg19_block5_conv1_conv2d_readvariableop_resource:J
;model_13_vgg19_block5_conv1_biasadd_readvariableop_resource:	V
:model_13_vgg19_block5_conv2_conv2d_readvariableop_resource:J
;model_13_vgg19_block5_conv2_biasadd_readvariableop_resource:	V
:model_13_vgg19_block5_conv3_conv2d_readvariableop_resource:J
;model_13_vgg19_block5_conv3_biasadd_readvariableop_resource:	V
:model_13_vgg19_block5_conv4_conv2d_readvariableop_resource:J
;model_13_vgg19_block5_conv4_biasadd_readvariableop_resource:	C
0model_13_dense_13_matmul_readvariableop_resource:	@?
1model_13_dense_13_biasadd_readvariableop_resource:@9
'dense_14_matmul_readvariableop_resource:6
(dense_14_biasadd_readvariableop_resource:
identity¢dense_14/BiasAdd/ReadVariableOp¢dense_14/MatMul/ReadVariableOp¢(model_13/dense_13/BiasAdd/ReadVariableOp¢*model_13/dense_13/BiasAdd_1/ReadVariableOp¢'model_13/dense_13/MatMul/ReadVariableOp¢)model_13/dense_13/MatMul_1/ReadVariableOp¢2model_13/vgg19/block1_conv1/BiasAdd/ReadVariableOp¢4model_13/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block1_conv1/Conv2D/ReadVariableOp¢3model_13/vgg19/block1_conv1/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block1_conv2/BiasAdd/ReadVariableOp¢4model_13/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block1_conv2/Conv2D/ReadVariableOp¢3model_13/vgg19/block1_conv2/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block2_conv1/BiasAdd/ReadVariableOp¢4model_13/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block2_conv1/Conv2D/ReadVariableOp¢3model_13/vgg19/block2_conv1/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block2_conv2/BiasAdd/ReadVariableOp¢4model_13/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block2_conv2/Conv2D/ReadVariableOp¢3model_13/vgg19/block2_conv2/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block3_conv1/BiasAdd/ReadVariableOp¢4model_13/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block3_conv1/Conv2D/ReadVariableOp¢3model_13/vgg19/block3_conv1/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block3_conv2/BiasAdd/ReadVariableOp¢4model_13/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block3_conv2/Conv2D/ReadVariableOp¢3model_13/vgg19/block3_conv2/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block3_conv3/BiasAdd/ReadVariableOp¢4model_13/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block3_conv3/Conv2D/ReadVariableOp¢3model_13/vgg19/block3_conv3/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block3_conv4/BiasAdd/ReadVariableOp¢4model_13/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block3_conv4/Conv2D/ReadVariableOp¢3model_13/vgg19/block3_conv4/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block4_conv1/BiasAdd/ReadVariableOp¢4model_13/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block4_conv1/Conv2D/ReadVariableOp¢3model_13/vgg19/block4_conv1/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block4_conv2/BiasAdd/ReadVariableOp¢4model_13/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block4_conv2/Conv2D/ReadVariableOp¢3model_13/vgg19/block4_conv2/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block4_conv3/BiasAdd/ReadVariableOp¢4model_13/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block4_conv3/Conv2D/ReadVariableOp¢3model_13/vgg19/block4_conv3/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block4_conv4/BiasAdd/ReadVariableOp¢4model_13/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block4_conv4/Conv2D/ReadVariableOp¢3model_13/vgg19/block4_conv4/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block5_conv1/BiasAdd/ReadVariableOp¢4model_13/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block5_conv1/Conv2D/ReadVariableOp¢3model_13/vgg19/block5_conv1/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block5_conv2/BiasAdd/ReadVariableOp¢4model_13/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block5_conv2/Conv2D/ReadVariableOp¢3model_13/vgg19/block5_conv2/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block5_conv3/BiasAdd/ReadVariableOp¢4model_13/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block5_conv3/Conv2D/ReadVariableOp¢3model_13/vgg19/block5_conv3/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block5_conv4/BiasAdd/ReadVariableOp¢4model_13/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block5_conv4/Conv2D/ReadVariableOp¢3model_13/vgg19/block5_conv4/Conv2D_1/ReadVariableOp´
1model_13/vgg19/block1_conv1/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Õ
"model_13/vgg19/block1_conv1/Conv2DConv2Dinputs_09model_13/vgg19/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
ª
2model_13/vgg19/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ó
#model_13/vgg19/block1_conv1/BiasAddBiasAdd+model_13/vgg19/block1_conv1/Conv2D:output:0:model_13/vgg19/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
 model_13/vgg19/block1_conv1/ReluRelu,model_13/vgg19/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@´
1model_13/vgg19/block1_conv2/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0û
"model_13/vgg19/block1_conv2/Conv2DConv2D.model_13/vgg19/block1_conv1/Relu:activations:09model_13/vgg19/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
ª
2model_13/vgg19/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ó
#model_13/vgg19/block1_conv2/BiasAddBiasAdd+model_13/vgg19/block1_conv2/Conv2D:output:0:model_13/vgg19/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
 model_13/vgg19/block1_conv2/ReluRelu,model_13/vgg19/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@Ê
"model_13/vgg19/block1_pool/MaxPoolMaxPool.model_13/vgg19/block1_conv2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@*
ksize
*
paddingVALID*
strides
µ
1model_13/vgg19/block2_conv1/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0÷
"model_13/vgg19/block2_conv1/Conv2DConv2D+model_13/vgg19/block1_pool/MaxPool:output:09model_13/vgg19/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
«
2model_13/vgg19/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block2_conv1/BiasAddBiasAdd+model_13/vgg19/block2_conv1/Conv2D:output:0:model_13/vgg19/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 model_13/vgg19/block2_conv1/ReluRelu,model_13/vgg19/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd¶
1model_13/vgg19/block2_conv2/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
"model_13/vgg19/block2_conv2/Conv2DConv2D.model_13/vgg19/block2_conv1/Relu:activations:09model_13/vgg19/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
«
2model_13/vgg19/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block2_conv2/BiasAddBiasAdd+model_13/vgg19/block2_conv2/Conv2D:output:0:model_13/vgg19/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 model_13/vgg19/block2_conv2/ReluRelu,model_13/vgg19/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddË
"model_13/vgg19/block2_pool/MaxPoolMaxPool.model_13/vgg19/block2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
ksize
*
paddingVALID*
strides
¶
1model_13/vgg19/block3_conv1/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0÷
"model_13/vgg19/block3_conv1/Conv2DConv2D+model_13/vgg19/block2_pool/MaxPool:output:09model_13/vgg19/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
«
2model_13/vgg19/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block3_conv1/BiasAddBiasAdd+model_13/vgg19/block3_conv1/Conv2D:output:0:model_13/vgg19/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 model_13/vgg19/block3_conv1/ReluRelu,model_13/vgg19/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¶
1model_13/vgg19/block3_conv2/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
"model_13/vgg19/block3_conv2/Conv2DConv2D.model_13/vgg19/block3_conv1/Relu:activations:09model_13/vgg19/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
«
2model_13/vgg19/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block3_conv2/BiasAddBiasAdd+model_13/vgg19/block3_conv2/Conv2D:output:0:model_13/vgg19/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 model_13/vgg19/block3_conv2/ReluRelu,model_13/vgg19/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¶
1model_13/vgg19/block3_conv3/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
"model_13/vgg19/block3_conv3/Conv2DConv2D.model_13/vgg19/block3_conv2/Relu:activations:09model_13/vgg19/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
«
2model_13/vgg19/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block3_conv3/BiasAddBiasAdd+model_13/vgg19/block3_conv3/Conv2D:output:0:model_13/vgg19/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 model_13/vgg19/block3_conv3/ReluRelu,model_13/vgg19/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¶
1model_13/vgg19/block3_conv4/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
"model_13/vgg19/block3_conv4/Conv2DConv2D.model_13/vgg19/block3_conv3/Relu:activations:09model_13/vgg19/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
«
2model_13/vgg19/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block3_conv4/BiasAddBiasAdd+model_13/vgg19/block3_conv4/Conv2D:output:0:model_13/vgg19/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 model_13/vgg19/block3_conv4/ReluRelu,model_13/vgg19/block3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Ë
"model_13/vgg19/block3_pool/MaxPoolMaxPool.model_13/vgg19/block3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¶
1model_13/vgg19/block4_conv1/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0÷
"model_13/vgg19/block4_conv1/Conv2DConv2D+model_13/vgg19/block3_pool/MaxPool:output:09model_13/vgg19/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
2model_13/vgg19/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block4_conv1/BiasAddBiasAdd+model_13/vgg19/block4_conv1/Conv2D:output:0:model_13/vgg19/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 model_13/vgg19/block4_conv1/ReluRelu,model_13/vgg19/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
1model_13/vgg19/block4_conv2/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
"model_13/vgg19/block4_conv2/Conv2DConv2D.model_13/vgg19/block4_conv1/Relu:activations:09model_13/vgg19/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
2model_13/vgg19/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block4_conv2/BiasAddBiasAdd+model_13/vgg19/block4_conv2/Conv2D:output:0:model_13/vgg19/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 model_13/vgg19/block4_conv2/ReluRelu,model_13/vgg19/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
1model_13/vgg19/block4_conv3/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
"model_13/vgg19/block4_conv3/Conv2DConv2D.model_13/vgg19/block4_conv2/Relu:activations:09model_13/vgg19/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
2model_13/vgg19/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block4_conv3/BiasAddBiasAdd+model_13/vgg19/block4_conv3/Conv2D:output:0:model_13/vgg19/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 model_13/vgg19/block4_conv3/ReluRelu,model_13/vgg19/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
1model_13/vgg19/block4_conv4/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
"model_13/vgg19/block4_conv4/Conv2DConv2D.model_13/vgg19/block4_conv3/Relu:activations:09model_13/vgg19/block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
2model_13/vgg19/block4_conv4/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block4_conv4/BiasAddBiasAdd+model_13/vgg19/block4_conv4/Conv2D:output:0:model_13/vgg19/block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 model_13/vgg19/block4_conv4/ReluRelu,model_13/vgg19/block4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
"model_13/vgg19/block4_pool/MaxPoolMaxPool.model_13/vgg19/block4_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¶
1model_13/vgg19/block5_conv1/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0÷
"model_13/vgg19/block5_conv1/Conv2DConv2D+model_13/vgg19/block4_pool/MaxPool:output:09model_13/vgg19/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
2model_13/vgg19/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block5_conv1/BiasAddBiasAdd+model_13/vgg19/block5_conv1/Conv2D:output:0:model_13/vgg19/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 model_13/vgg19/block5_conv1/ReluRelu,model_13/vgg19/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
1model_13/vgg19/block5_conv2/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
"model_13/vgg19/block5_conv2/Conv2DConv2D.model_13/vgg19/block5_conv1/Relu:activations:09model_13/vgg19/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
2model_13/vgg19/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block5_conv2/BiasAddBiasAdd+model_13/vgg19/block5_conv2/Conv2D:output:0:model_13/vgg19/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 model_13/vgg19/block5_conv2/ReluRelu,model_13/vgg19/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
1model_13/vgg19/block5_conv3/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
"model_13/vgg19/block5_conv3/Conv2DConv2D.model_13/vgg19/block5_conv2/Relu:activations:09model_13/vgg19/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
2model_13/vgg19/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block5_conv3/BiasAddBiasAdd+model_13/vgg19/block5_conv3/Conv2D:output:0:model_13/vgg19/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 model_13/vgg19/block5_conv3/ReluRelu,model_13/vgg19/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
1model_13/vgg19/block5_conv4/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
"model_13/vgg19/block5_conv4/Conv2DConv2D.model_13/vgg19/block5_conv3/Relu:activations:09model_13/vgg19/block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
2model_13/vgg19/block5_conv4/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block5_conv4/BiasAddBiasAdd+model_13/vgg19/block5_conv4/Conv2D:output:0:model_13/vgg19/block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 model_13/vgg19/block5_conv4/ReluRelu,model_13/vgg19/block5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
"model_13/vgg19/block5_pool/MaxPoolMaxPool.model_13/vgg19/block5_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

:model_13/global_average_pooling2d_9/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Õ
(model_13/global_average_pooling2d_9/MeanMean+model_13/vgg19/block5_pool/MaxPool:output:0Cmodel_13/global_average_pooling2d_9/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model_13/dense_13/MatMul/ReadVariableOpReadVariableOp0model_13_dense_13_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0¸
model_13/dense_13/MatMulMatMul1model_13/global_average_pooling2d_9/Mean:output:0/model_13/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(model_13/dense_13/BiasAdd/ReadVariableOpReadVariableOp1model_13_dense_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¬
model_13/dense_13/BiasAddBiasAdd"model_13/dense_13/MatMul:product:00model_13/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¶
3model_13/vgg19/block1_conv1/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ù
$model_13/vgg19/block1_conv1/Conv2D_1Conv2Dinputs_1;model_13/vgg19/block1_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
¬
4model_13/vgg19/block1_conv1/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ù
%model_13/vgg19/block1_conv1/BiasAdd_1BiasAdd-model_13/vgg19/block1_conv1/Conv2D_1:output:0<model_13/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
"model_13/vgg19/block1_conv1/Relu_1Relu.model_13/vgg19/block1_conv1/BiasAdd_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@¶
3model_13/vgg19/block1_conv2/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
$model_13/vgg19/block1_conv2/Conv2D_1Conv2D0model_13/vgg19/block1_conv1/Relu_1:activations:0;model_13/vgg19/block1_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
¬
4model_13/vgg19/block1_conv2/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ù
%model_13/vgg19/block1_conv2/BiasAdd_1BiasAdd-model_13/vgg19/block1_conv2/Conv2D_1:output:0<model_13/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
"model_13/vgg19/block1_conv2/Relu_1Relu.model_13/vgg19/block1_conv2/BiasAdd_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@Î
$model_13/vgg19/block1_pool/MaxPool_1MaxPool0model_13/vgg19/block1_conv2/Relu_1:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@*
ksize
*
paddingVALID*
strides
·
3model_13/vgg19/block2_conv1/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ý
$model_13/vgg19/block2_conv1/Conv2D_1Conv2D-model_13/vgg19/block1_pool/MaxPool_1:output:0;model_13/vgg19/block2_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
­
4model_13/vgg19/block2_conv1/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block2_conv1/BiasAdd_1BiasAdd-model_13/vgg19/block2_conv1/Conv2D_1:output:0<model_13/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
"model_13/vgg19/block2_conv1/Relu_1Relu.model_13/vgg19/block2_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd¸
3model_13/vgg19/block2_conv2/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$model_13/vgg19/block2_conv2/Conv2D_1Conv2D0model_13/vgg19/block2_conv1/Relu_1:activations:0;model_13/vgg19/block2_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
­
4model_13/vgg19/block2_conv2/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block2_conv2/BiasAdd_1BiasAdd-model_13/vgg19/block2_conv2/Conv2D_1:output:0<model_13/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
"model_13/vgg19/block2_conv2/Relu_1Relu.model_13/vgg19/block2_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddÏ
$model_13/vgg19/block2_pool/MaxPool_1MaxPool0model_13/vgg19/block2_conv2/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
ksize
*
paddingVALID*
strides
¸
3model_13/vgg19/block3_conv1/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
$model_13/vgg19/block3_conv1/Conv2D_1Conv2D-model_13/vgg19/block2_pool/MaxPool_1:output:0;model_13/vgg19/block3_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
­
4model_13/vgg19/block3_conv1/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block3_conv1/BiasAdd_1BiasAdd-model_13/vgg19/block3_conv1/Conv2D_1:output:0<model_13/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
"model_13/vgg19/block3_conv1/Relu_1Relu.model_13/vgg19/block3_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¸
3model_13/vgg19/block3_conv2/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$model_13/vgg19/block3_conv2/Conv2D_1Conv2D0model_13/vgg19/block3_conv1/Relu_1:activations:0;model_13/vgg19/block3_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
­
4model_13/vgg19/block3_conv2/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block3_conv2/BiasAdd_1BiasAdd-model_13/vgg19/block3_conv2/Conv2D_1:output:0<model_13/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
"model_13/vgg19/block3_conv2/Relu_1Relu.model_13/vgg19/block3_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¸
3model_13/vgg19/block3_conv3/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$model_13/vgg19/block3_conv3/Conv2D_1Conv2D0model_13/vgg19/block3_conv2/Relu_1:activations:0;model_13/vgg19/block3_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
­
4model_13/vgg19/block3_conv3/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block3_conv3/BiasAdd_1BiasAdd-model_13/vgg19/block3_conv3/Conv2D_1:output:0<model_13/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
"model_13/vgg19/block3_conv3/Relu_1Relu.model_13/vgg19/block3_conv3/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¸
3model_13/vgg19/block3_conv4/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$model_13/vgg19/block3_conv4/Conv2D_1Conv2D0model_13/vgg19/block3_conv3/Relu_1:activations:0;model_13/vgg19/block3_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
­
4model_13/vgg19/block3_conv4/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block3_conv4/BiasAdd_1BiasAdd-model_13/vgg19/block3_conv4/Conv2D_1:output:0<model_13/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
"model_13/vgg19/block3_conv4/Relu_1Relu.model_13/vgg19/block3_conv4/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Ï
$model_13/vgg19/block3_pool/MaxPool_1MaxPool0model_13/vgg19/block3_conv4/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¸
3model_13/vgg19/block4_conv1/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
$model_13/vgg19/block4_conv1/Conv2D_1Conv2D-model_13/vgg19/block3_pool/MaxPool_1:output:0;model_13/vgg19/block4_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
­
4model_13/vgg19/block4_conv1/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block4_conv1/BiasAdd_1BiasAdd-model_13/vgg19/block4_conv1/Conv2D_1:output:0<model_13/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"model_13/vgg19/block4_conv1/Relu_1Relu.model_13/vgg19/block4_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
3model_13/vgg19/block4_conv2/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$model_13/vgg19/block4_conv2/Conv2D_1Conv2D0model_13/vgg19/block4_conv1/Relu_1:activations:0;model_13/vgg19/block4_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
­
4model_13/vgg19/block4_conv2/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block4_conv2/BiasAdd_1BiasAdd-model_13/vgg19/block4_conv2/Conv2D_1:output:0<model_13/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"model_13/vgg19/block4_conv2/Relu_1Relu.model_13/vgg19/block4_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
3model_13/vgg19/block4_conv3/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$model_13/vgg19/block4_conv3/Conv2D_1Conv2D0model_13/vgg19/block4_conv2/Relu_1:activations:0;model_13/vgg19/block4_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
­
4model_13/vgg19/block4_conv3/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block4_conv3/BiasAdd_1BiasAdd-model_13/vgg19/block4_conv3/Conv2D_1:output:0<model_13/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"model_13/vgg19/block4_conv3/Relu_1Relu.model_13/vgg19/block4_conv3/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
3model_13/vgg19/block4_conv4/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$model_13/vgg19/block4_conv4/Conv2D_1Conv2D0model_13/vgg19/block4_conv3/Relu_1:activations:0;model_13/vgg19/block4_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
­
4model_13/vgg19/block4_conv4/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block4_conv4/BiasAdd_1BiasAdd-model_13/vgg19/block4_conv4/Conv2D_1:output:0<model_13/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"model_13/vgg19/block4_conv4/Relu_1Relu.model_13/vgg19/block4_conv4/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
$model_13/vgg19/block4_pool/MaxPool_1MaxPool0model_13/vgg19/block4_conv4/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¸
3model_13/vgg19/block5_conv1/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
$model_13/vgg19/block5_conv1/Conv2D_1Conv2D-model_13/vgg19/block4_pool/MaxPool_1:output:0;model_13/vgg19/block5_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
­
4model_13/vgg19/block5_conv1/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block5_conv1/BiasAdd_1BiasAdd-model_13/vgg19/block5_conv1/Conv2D_1:output:0<model_13/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"model_13/vgg19/block5_conv1/Relu_1Relu.model_13/vgg19/block5_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
3model_13/vgg19/block5_conv2/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$model_13/vgg19/block5_conv2/Conv2D_1Conv2D0model_13/vgg19/block5_conv1/Relu_1:activations:0;model_13/vgg19/block5_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
­
4model_13/vgg19/block5_conv2/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block5_conv2/BiasAdd_1BiasAdd-model_13/vgg19/block5_conv2/Conv2D_1:output:0<model_13/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"model_13/vgg19/block5_conv2/Relu_1Relu.model_13/vgg19/block5_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
3model_13/vgg19/block5_conv3/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$model_13/vgg19/block5_conv3/Conv2D_1Conv2D0model_13/vgg19/block5_conv2/Relu_1:activations:0;model_13/vgg19/block5_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
­
4model_13/vgg19/block5_conv3/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block5_conv3/BiasAdd_1BiasAdd-model_13/vgg19/block5_conv3/Conv2D_1:output:0<model_13/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"model_13/vgg19/block5_conv3/Relu_1Relu.model_13/vgg19/block5_conv3/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
3model_13/vgg19/block5_conv4/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$model_13/vgg19/block5_conv4/Conv2D_1Conv2D0model_13/vgg19/block5_conv3/Relu_1:activations:0;model_13/vgg19/block5_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
­
4model_13/vgg19/block5_conv4/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block5_conv4/BiasAdd_1BiasAdd-model_13/vgg19/block5_conv4/Conv2D_1:output:0<model_13/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"model_13/vgg19/block5_conv4/Relu_1Relu.model_13/vgg19/block5_conv4/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
$model_13/vgg19/block5_pool/MaxPool_1MaxPool0model_13/vgg19/block5_conv4/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

<model_13/global_average_pooling2d_9/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Û
*model_13/global_average_pooling2d_9/Mean_1Mean-model_13/vgg19/block5_pool/MaxPool_1:output:0Emodel_13/global_average_pooling2d_9/Mean_1/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)model_13/dense_13/MatMul_1/ReadVariableOpReadVariableOp0model_13_dense_13_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0¾
model_13/dense_13/MatMul_1MatMul3model_13/global_average_pooling2d_9/Mean_1:output:01model_13/dense_13/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*model_13/dense_13/BiasAdd_1/ReadVariableOpReadVariableOp1model_13_dense_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0²
model_13/dense_13/BiasAdd_1BiasAdd$model_13/dense_13/MatMul_1:product:02model_13/dense_13/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lambda_4/subSub"model_13/dense_13/BiasAdd:output:0$model_13/dense_13/BiasAdd_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
lambda_4/SquareSquarelambda_4/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
lambda_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
lambda_4/SumSumlambda_4/Square:y:0'lambda_4/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(W
lambda_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
lambda_4/MaximumMaximumlambda_4/Sum:output:0lambda_4/Maximum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
lambda_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
lambda_4/Maximum_1Maximumlambda_4/Maximum:z:0lambda_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lambda_4/SqrtSqrtlambda_4/Maximum_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_14/MatMulMatMullambda_4/Sqrt:y:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_14/SigmoidSigmoiddense_14/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitydense_14/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp)^model_13/dense_13/BiasAdd/ReadVariableOp+^model_13/dense_13/BiasAdd_1/ReadVariableOp(^model_13/dense_13/MatMul/ReadVariableOp*^model_13/dense_13/MatMul_1/ReadVariableOp3^model_13/vgg19/block1_conv1/BiasAdd/ReadVariableOp5^model_13/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block1_conv1/Conv2D/ReadVariableOp4^model_13/vgg19/block1_conv1/Conv2D_1/ReadVariableOp3^model_13/vgg19/block1_conv2/BiasAdd/ReadVariableOp5^model_13/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block1_conv2/Conv2D/ReadVariableOp4^model_13/vgg19/block1_conv2/Conv2D_1/ReadVariableOp3^model_13/vgg19/block2_conv1/BiasAdd/ReadVariableOp5^model_13/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block2_conv1/Conv2D/ReadVariableOp4^model_13/vgg19/block2_conv1/Conv2D_1/ReadVariableOp3^model_13/vgg19/block2_conv2/BiasAdd/ReadVariableOp5^model_13/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block2_conv2/Conv2D/ReadVariableOp4^model_13/vgg19/block2_conv2/Conv2D_1/ReadVariableOp3^model_13/vgg19/block3_conv1/BiasAdd/ReadVariableOp5^model_13/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block3_conv1/Conv2D/ReadVariableOp4^model_13/vgg19/block3_conv1/Conv2D_1/ReadVariableOp3^model_13/vgg19/block3_conv2/BiasAdd/ReadVariableOp5^model_13/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block3_conv2/Conv2D/ReadVariableOp4^model_13/vgg19/block3_conv2/Conv2D_1/ReadVariableOp3^model_13/vgg19/block3_conv3/BiasAdd/ReadVariableOp5^model_13/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block3_conv3/Conv2D/ReadVariableOp4^model_13/vgg19/block3_conv3/Conv2D_1/ReadVariableOp3^model_13/vgg19/block3_conv4/BiasAdd/ReadVariableOp5^model_13/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block3_conv4/Conv2D/ReadVariableOp4^model_13/vgg19/block3_conv4/Conv2D_1/ReadVariableOp3^model_13/vgg19/block4_conv1/BiasAdd/ReadVariableOp5^model_13/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block4_conv1/Conv2D/ReadVariableOp4^model_13/vgg19/block4_conv1/Conv2D_1/ReadVariableOp3^model_13/vgg19/block4_conv2/BiasAdd/ReadVariableOp5^model_13/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block4_conv2/Conv2D/ReadVariableOp4^model_13/vgg19/block4_conv2/Conv2D_1/ReadVariableOp3^model_13/vgg19/block4_conv3/BiasAdd/ReadVariableOp5^model_13/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block4_conv3/Conv2D/ReadVariableOp4^model_13/vgg19/block4_conv3/Conv2D_1/ReadVariableOp3^model_13/vgg19/block4_conv4/BiasAdd/ReadVariableOp5^model_13/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block4_conv4/Conv2D/ReadVariableOp4^model_13/vgg19/block4_conv4/Conv2D_1/ReadVariableOp3^model_13/vgg19/block5_conv1/BiasAdd/ReadVariableOp5^model_13/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block5_conv1/Conv2D/ReadVariableOp4^model_13/vgg19/block5_conv1/Conv2D_1/ReadVariableOp3^model_13/vgg19/block5_conv2/BiasAdd/ReadVariableOp5^model_13/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block5_conv2/Conv2D/ReadVariableOp4^model_13/vgg19/block5_conv2/Conv2D_1/ReadVariableOp3^model_13/vgg19/block5_conv3/BiasAdd/ReadVariableOp5^model_13/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block5_conv3/Conv2D/ReadVariableOp4^model_13/vgg19/block5_conv3/Conv2D_1/ReadVariableOp3^model_13/vgg19/block5_conv4/BiasAdd/ReadVariableOp5^model_13/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block5_conv4/Conv2D/ReadVariableOp4^model_13/vgg19/block5_conv4/Conv2D_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2T
(model_13/dense_13/BiasAdd/ReadVariableOp(model_13/dense_13/BiasAdd/ReadVariableOp2X
*model_13/dense_13/BiasAdd_1/ReadVariableOp*model_13/dense_13/BiasAdd_1/ReadVariableOp2R
'model_13/dense_13/MatMul/ReadVariableOp'model_13/dense_13/MatMul/ReadVariableOp2V
)model_13/dense_13/MatMul_1/ReadVariableOp)model_13/dense_13/MatMul_1/ReadVariableOp2h
2model_13/vgg19/block1_conv1/BiasAdd/ReadVariableOp2model_13/vgg19/block1_conv1/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp4model_13/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block1_conv1/Conv2D/ReadVariableOp1model_13/vgg19/block1_conv1/Conv2D/ReadVariableOp2j
3model_13/vgg19/block1_conv1/Conv2D_1/ReadVariableOp3model_13/vgg19/block1_conv1/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block1_conv2/BiasAdd/ReadVariableOp2model_13/vgg19/block1_conv2/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp4model_13/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block1_conv2/Conv2D/ReadVariableOp1model_13/vgg19/block1_conv2/Conv2D/ReadVariableOp2j
3model_13/vgg19/block1_conv2/Conv2D_1/ReadVariableOp3model_13/vgg19/block1_conv2/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block2_conv1/BiasAdd/ReadVariableOp2model_13/vgg19/block2_conv1/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp4model_13/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block2_conv1/Conv2D/ReadVariableOp1model_13/vgg19/block2_conv1/Conv2D/ReadVariableOp2j
3model_13/vgg19/block2_conv1/Conv2D_1/ReadVariableOp3model_13/vgg19/block2_conv1/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block2_conv2/BiasAdd/ReadVariableOp2model_13/vgg19/block2_conv2/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp4model_13/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block2_conv2/Conv2D/ReadVariableOp1model_13/vgg19/block2_conv2/Conv2D/ReadVariableOp2j
3model_13/vgg19/block2_conv2/Conv2D_1/ReadVariableOp3model_13/vgg19/block2_conv2/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block3_conv1/BiasAdd/ReadVariableOp2model_13/vgg19/block3_conv1/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp4model_13/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block3_conv1/Conv2D/ReadVariableOp1model_13/vgg19/block3_conv1/Conv2D/ReadVariableOp2j
3model_13/vgg19/block3_conv1/Conv2D_1/ReadVariableOp3model_13/vgg19/block3_conv1/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block3_conv2/BiasAdd/ReadVariableOp2model_13/vgg19/block3_conv2/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp4model_13/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block3_conv2/Conv2D/ReadVariableOp1model_13/vgg19/block3_conv2/Conv2D/ReadVariableOp2j
3model_13/vgg19/block3_conv2/Conv2D_1/ReadVariableOp3model_13/vgg19/block3_conv2/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block3_conv3/BiasAdd/ReadVariableOp2model_13/vgg19/block3_conv3/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp4model_13/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block3_conv3/Conv2D/ReadVariableOp1model_13/vgg19/block3_conv3/Conv2D/ReadVariableOp2j
3model_13/vgg19/block3_conv3/Conv2D_1/ReadVariableOp3model_13/vgg19/block3_conv3/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block3_conv4/BiasAdd/ReadVariableOp2model_13/vgg19/block3_conv4/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp4model_13/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block3_conv4/Conv2D/ReadVariableOp1model_13/vgg19/block3_conv4/Conv2D/ReadVariableOp2j
3model_13/vgg19/block3_conv4/Conv2D_1/ReadVariableOp3model_13/vgg19/block3_conv4/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block4_conv1/BiasAdd/ReadVariableOp2model_13/vgg19/block4_conv1/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp4model_13/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block4_conv1/Conv2D/ReadVariableOp1model_13/vgg19/block4_conv1/Conv2D/ReadVariableOp2j
3model_13/vgg19/block4_conv1/Conv2D_1/ReadVariableOp3model_13/vgg19/block4_conv1/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block4_conv2/BiasAdd/ReadVariableOp2model_13/vgg19/block4_conv2/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp4model_13/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block4_conv2/Conv2D/ReadVariableOp1model_13/vgg19/block4_conv2/Conv2D/ReadVariableOp2j
3model_13/vgg19/block4_conv2/Conv2D_1/ReadVariableOp3model_13/vgg19/block4_conv2/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block4_conv3/BiasAdd/ReadVariableOp2model_13/vgg19/block4_conv3/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp4model_13/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block4_conv3/Conv2D/ReadVariableOp1model_13/vgg19/block4_conv3/Conv2D/ReadVariableOp2j
3model_13/vgg19/block4_conv3/Conv2D_1/ReadVariableOp3model_13/vgg19/block4_conv3/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block4_conv4/BiasAdd/ReadVariableOp2model_13/vgg19/block4_conv4/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp4model_13/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block4_conv4/Conv2D/ReadVariableOp1model_13/vgg19/block4_conv4/Conv2D/ReadVariableOp2j
3model_13/vgg19/block4_conv4/Conv2D_1/ReadVariableOp3model_13/vgg19/block4_conv4/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block5_conv1/BiasAdd/ReadVariableOp2model_13/vgg19/block5_conv1/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp4model_13/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block5_conv1/Conv2D/ReadVariableOp1model_13/vgg19/block5_conv1/Conv2D/ReadVariableOp2j
3model_13/vgg19/block5_conv1/Conv2D_1/ReadVariableOp3model_13/vgg19/block5_conv1/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block5_conv2/BiasAdd/ReadVariableOp2model_13/vgg19/block5_conv2/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp4model_13/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block5_conv2/Conv2D/ReadVariableOp1model_13/vgg19/block5_conv2/Conv2D/ReadVariableOp2j
3model_13/vgg19/block5_conv2/Conv2D_1/ReadVariableOp3model_13/vgg19/block5_conv2/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block5_conv3/BiasAdd/ReadVariableOp2model_13/vgg19/block5_conv3/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp4model_13/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block5_conv3/Conv2D/ReadVariableOp1model_13/vgg19/block5_conv3/Conv2D/ReadVariableOp2j
3model_13/vgg19/block5_conv3/Conv2D_1/ReadVariableOp3model_13/vgg19/block5_conv3/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block5_conv4/BiasAdd/ReadVariableOp2model_13/vgg19/block5_conv4/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp4model_13/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block5_conv4/Conv2D/ReadVariableOp1model_13/vgg19/block5_conv4/Conv2D/ReadVariableOp2j
3model_13/vgg19/block5_conv4/Conv2D_1/ReadVariableOp3model_13/vgg19/block5_conv4/Conv2D_1/ReadVariableOp:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
inputs/1
)
ª
D__inference_model_14_layer_call_and_return_conditional_losses_976885

inputs
inputs_1)
model_13_976774:@
model_13_976776:@)
model_13_976778:@@
model_13_976780:@*
model_13_976782:@
model_13_976784:	+
model_13_976786:
model_13_976788:	+
model_13_976790:
model_13_976792:	+
model_13_976794:
model_13_976796:	+
model_13_976798:
model_13_976800:	+
model_13_976802:
model_13_976804:	+
model_13_976806:
model_13_976808:	+
model_13_976810:
model_13_976812:	+
model_13_976814:
model_13_976816:	+
model_13_976818:
model_13_976820:	+
model_13_976822:
model_13_976824:	+
model_13_976826:
model_13_976828:	+
model_13_976830:
model_13_976832:	+
model_13_976834:
model_13_976836:	"
model_13_976838:	@
model_13_976840:@!
dense_14_976879:
dense_14_976881:
identity¢ dense_14/StatefulPartitionedCall¢ model_13/StatefulPartitionedCall¢"model_13/StatefulPartitionedCall_1Ó
 model_13/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_13_976774model_13_976776model_13_976778model_13_976780model_13_976782model_13_976784model_13_976786model_13_976788model_13_976790model_13_976792model_13_976794model_13_976796model_13_976798model_13_976800model_13_976802model_13_976804model_13_976806model_13_976808model_13_976810model_13_976812model_13_976814model_13_976816model_13_976818model_13_976820model_13_976822model_13_976824model_13_976826model_13_976828model_13_976830model_13_976832model_13_976834model_13_976836model_13_976838model_13_976840*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_13_layer_call_and_return_conditional_losses_976136×
"model_13/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1model_13_976774model_13_976776model_13_976778model_13_976780model_13_976782model_13_976784model_13_976786model_13_976788model_13_976790model_13_976792model_13_976794model_13_976796model_13_976798model_13_976800model_13_976802model_13_976804model_13_976806model_13_976808model_13_976810model_13_976812model_13_976814model_13_976816model_13_976818model_13_976820model_13_976822model_13_976824model_13_976826model_13_976828model_13_976830model_13_976832model_13_976834model_13_976836model_13_976838model_13_976840*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_13_layer_call_and_return_conditional_losses_976136
lambda_4/PartitionedCallPartitionedCall)model_13/StatefulPartitionedCall:output:0+model_13/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_lambda_4_layer_call_and_return_conditional_losses_976686
 dense_14/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0dense_14_976879dense_14_976881*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_976570x
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
NoOpNoOp!^dense_14/StatefulPartitionedCall!^model_13/StatefulPartitionedCall#^model_13/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 model_13/StatefulPartitionedCall model_13/StatefulPartitionedCall2H
"model_13/StatefulPartitionedCall_1"model_13/StatefulPartitionedCall_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs:YU
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs

c
G__inference_block2_pool_layer_call_and_return_conditional_losses_974777

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
¥
-__inference_block5_conv2_layer_call_fn_979165

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv2_layer_call_and_return_conditional_losses_975059x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±
ö3
D__inference_model_14_layer_call_and_return_conditional_losses_977902
inputs_0
inputs_1T
:model_13_vgg19_block1_conv1_conv2d_readvariableop_resource:@I
;model_13_vgg19_block1_conv1_biasadd_readvariableop_resource:@T
:model_13_vgg19_block1_conv2_conv2d_readvariableop_resource:@@I
;model_13_vgg19_block1_conv2_biasadd_readvariableop_resource:@U
:model_13_vgg19_block2_conv1_conv2d_readvariableop_resource:@J
;model_13_vgg19_block2_conv1_biasadd_readvariableop_resource:	V
:model_13_vgg19_block2_conv2_conv2d_readvariableop_resource:J
;model_13_vgg19_block2_conv2_biasadd_readvariableop_resource:	V
:model_13_vgg19_block3_conv1_conv2d_readvariableop_resource:J
;model_13_vgg19_block3_conv1_biasadd_readvariableop_resource:	V
:model_13_vgg19_block3_conv2_conv2d_readvariableop_resource:J
;model_13_vgg19_block3_conv2_biasadd_readvariableop_resource:	V
:model_13_vgg19_block3_conv3_conv2d_readvariableop_resource:J
;model_13_vgg19_block3_conv3_biasadd_readvariableop_resource:	V
:model_13_vgg19_block3_conv4_conv2d_readvariableop_resource:J
;model_13_vgg19_block3_conv4_biasadd_readvariableop_resource:	V
:model_13_vgg19_block4_conv1_conv2d_readvariableop_resource:J
;model_13_vgg19_block4_conv1_biasadd_readvariableop_resource:	V
:model_13_vgg19_block4_conv2_conv2d_readvariableop_resource:J
;model_13_vgg19_block4_conv2_biasadd_readvariableop_resource:	V
:model_13_vgg19_block4_conv3_conv2d_readvariableop_resource:J
;model_13_vgg19_block4_conv3_biasadd_readvariableop_resource:	V
:model_13_vgg19_block4_conv4_conv2d_readvariableop_resource:J
;model_13_vgg19_block4_conv4_biasadd_readvariableop_resource:	V
:model_13_vgg19_block5_conv1_conv2d_readvariableop_resource:J
;model_13_vgg19_block5_conv1_biasadd_readvariableop_resource:	V
:model_13_vgg19_block5_conv2_conv2d_readvariableop_resource:J
;model_13_vgg19_block5_conv2_biasadd_readvariableop_resource:	V
:model_13_vgg19_block5_conv3_conv2d_readvariableop_resource:J
;model_13_vgg19_block5_conv3_biasadd_readvariableop_resource:	V
:model_13_vgg19_block5_conv4_conv2d_readvariableop_resource:J
;model_13_vgg19_block5_conv4_biasadd_readvariableop_resource:	C
0model_13_dense_13_matmul_readvariableop_resource:	@?
1model_13_dense_13_biasadd_readvariableop_resource:@9
'dense_14_matmul_readvariableop_resource:6
(dense_14_biasadd_readvariableop_resource:
identity¢dense_14/BiasAdd/ReadVariableOp¢dense_14/MatMul/ReadVariableOp¢(model_13/dense_13/BiasAdd/ReadVariableOp¢*model_13/dense_13/BiasAdd_1/ReadVariableOp¢'model_13/dense_13/MatMul/ReadVariableOp¢)model_13/dense_13/MatMul_1/ReadVariableOp¢2model_13/vgg19/block1_conv1/BiasAdd/ReadVariableOp¢4model_13/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block1_conv1/Conv2D/ReadVariableOp¢3model_13/vgg19/block1_conv1/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block1_conv2/BiasAdd/ReadVariableOp¢4model_13/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block1_conv2/Conv2D/ReadVariableOp¢3model_13/vgg19/block1_conv2/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block2_conv1/BiasAdd/ReadVariableOp¢4model_13/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block2_conv1/Conv2D/ReadVariableOp¢3model_13/vgg19/block2_conv1/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block2_conv2/BiasAdd/ReadVariableOp¢4model_13/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block2_conv2/Conv2D/ReadVariableOp¢3model_13/vgg19/block2_conv2/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block3_conv1/BiasAdd/ReadVariableOp¢4model_13/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block3_conv1/Conv2D/ReadVariableOp¢3model_13/vgg19/block3_conv1/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block3_conv2/BiasAdd/ReadVariableOp¢4model_13/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block3_conv2/Conv2D/ReadVariableOp¢3model_13/vgg19/block3_conv2/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block3_conv3/BiasAdd/ReadVariableOp¢4model_13/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block3_conv3/Conv2D/ReadVariableOp¢3model_13/vgg19/block3_conv3/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block3_conv4/BiasAdd/ReadVariableOp¢4model_13/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block3_conv4/Conv2D/ReadVariableOp¢3model_13/vgg19/block3_conv4/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block4_conv1/BiasAdd/ReadVariableOp¢4model_13/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block4_conv1/Conv2D/ReadVariableOp¢3model_13/vgg19/block4_conv1/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block4_conv2/BiasAdd/ReadVariableOp¢4model_13/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block4_conv2/Conv2D/ReadVariableOp¢3model_13/vgg19/block4_conv2/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block4_conv3/BiasAdd/ReadVariableOp¢4model_13/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block4_conv3/Conv2D/ReadVariableOp¢3model_13/vgg19/block4_conv3/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block4_conv4/BiasAdd/ReadVariableOp¢4model_13/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block4_conv4/Conv2D/ReadVariableOp¢3model_13/vgg19/block4_conv4/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block5_conv1/BiasAdd/ReadVariableOp¢4model_13/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block5_conv1/Conv2D/ReadVariableOp¢3model_13/vgg19/block5_conv1/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block5_conv2/BiasAdd/ReadVariableOp¢4model_13/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block5_conv2/Conv2D/ReadVariableOp¢3model_13/vgg19/block5_conv2/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block5_conv3/BiasAdd/ReadVariableOp¢4model_13/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block5_conv3/Conv2D/ReadVariableOp¢3model_13/vgg19/block5_conv3/Conv2D_1/ReadVariableOp¢2model_13/vgg19/block5_conv4/BiasAdd/ReadVariableOp¢4model_13/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp¢1model_13/vgg19/block5_conv4/Conv2D/ReadVariableOp¢3model_13/vgg19/block5_conv4/Conv2D_1/ReadVariableOp´
1model_13/vgg19/block1_conv1/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Õ
"model_13/vgg19/block1_conv1/Conv2DConv2Dinputs_09model_13/vgg19/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
ª
2model_13/vgg19/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ó
#model_13/vgg19/block1_conv1/BiasAddBiasAdd+model_13/vgg19/block1_conv1/Conv2D:output:0:model_13/vgg19/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
 model_13/vgg19/block1_conv1/ReluRelu,model_13/vgg19/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@´
1model_13/vgg19/block1_conv2/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0û
"model_13/vgg19/block1_conv2/Conv2DConv2D.model_13/vgg19/block1_conv1/Relu:activations:09model_13/vgg19/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
ª
2model_13/vgg19/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ó
#model_13/vgg19/block1_conv2/BiasAddBiasAdd+model_13/vgg19/block1_conv2/Conv2D:output:0:model_13/vgg19/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
 model_13/vgg19/block1_conv2/ReluRelu,model_13/vgg19/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@Ê
"model_13/vgg19/block1_pool/MaxPoolMaxPool.model_13/vgg19/block1_conv2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@*
ksize
*
paddingVALID*
strides
µ
1model_13/vgg19/block2_conv1/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0÷
"model_13/vgg19/block2_conv1/Conv2DConv2D+model_13/vgg19/block1_pool/MaxPool:output:09model_13/vgg19/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
«
2model_13/vgg19/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block2_conv1/BiasAddBiasAdd+model_13/vgg19/block2_conv1/Conv2D:output:0:model_13/vgg19/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 model_13/vgg19/block2_conv1/ReluRelu,model_13/vgg19/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd¶
1model_13/vgg19/block2_conv2/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
"model_13/vgg19/block2_conv2/Conv2DConv2D.model_13/vgg19/block2_conv1/Relu:activations:09model_13/vgg19/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
«
2model_13/vgg19/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block2_conv2/BiasAddBiasAdd+model_13/vgg19/block2_conv2/Conv2D:output:0:model_13/vgg19/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 model_13/vgg19/block2_conv2/ReluRelu,model_13/vgg19/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddË
"model_13/vgg19/block2_pool/MaxPoolMaxPool.model_13/vgg19/block2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
ksize
*
paddingVALID*
strides
¶
1model_13/vgg19/block3_conv1/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0÷
"model_13/vgg19/block3_conv1/Conv2DConv2D+model_13/vgg19/block2_pool/MaxPool:output:09model_13/vgg19/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
«
2model_13/vgg19/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block3_conv1/BiasAddBiasAdd+model_13/vgg19/block3_conv1/Conv2D:output:0:model_13/vgg19/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 model_13/vgg19/block3_conv1/ReluRelu,model_13/vgg19/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¶
1model_13/vgg19/block3_conv2/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
"model_13/vgg19/block3_conv2/Conv2DConv2D.model_13/vgg19/block3_conv1/Relu:activations:09model_13/vgg19/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
«
2model_13/vgg19/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block3_conv2/BiasAddBiasAdd+model_13/vgg19/block3_conv2/Conv2D:output:0:model_13/vgg19/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 model_13/vgg19/block3_conv2/ReluRelu,model_13/vgg19/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¶
1model_13/vgg19/block3_conv3/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
"model_13/vgg19/block3_conv3/Conv2DConv2D.model_13/vgg19/block3_conv2/Relu:activations:09model_13/vgg19/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
«
2model_13/vgg19/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block3_conv3/BiasAddBiasAdd+model_13/vgg19/block3_conv3/Conv2D:output:0:model_13/vgg19/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 model_13/vgg19/block3_conv3/ReluRelu,model_13/vgg19/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¶
1model_13/vgg19/block3_conv4/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
"model_13/vgg19/block3_conv4/Conv2DConv2D.model_13/vgg19/block3_conv3/Relu:activations:09model_13/vgg19/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
«
2model_13/vgg19/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block3_conv4/BiasAddBiasAdd+model_13/vgg19/block3_conv4/Conv2D:output:0:model_13/vgg19/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 model_13/vgg19/block3_conv4/ReluRelu,model_13/vgg19/block3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Ë
"model_13/vgg19/block3_pool/MaxPoolMaxPool.model_13/vgg19/block3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¶
1model_13/vgg19/block4_conv1/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0÷
"model_13/vgg19/block4_conv1/Conv2DConv2D+model_13/vgg19/block3_pool/MaxPool:output:09model_13/vgg19/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
2model_13/vgg19/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block4_conv1/BiasAddBiasAdd+model_13/vgg19/block4_conv1/Conv2D:output:0:model_13/vgg19/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 model_13/vgg19/block4_conv1/ReluRelu,model_13/vgg19/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
1model_13/vgg19/block4_conv2/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
"model_13/vgg19/block4_conv2/Conv2DConv2D.model_13/vgg19/block4_conv1/Relu:activations:09model_13/vgg19/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
2model_13/vgg19/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block4_conv2/BiasAddBiasAdd+model_13/vgg19/block4_conv2/Conv2D:output:0:model_13/vgg19/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 model_13/vgg19/block4_conv2/ReluRelu,model_13/vgg19/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
1model_13/vgg19/block4_conv3/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
"model_13/vgg19/block4_conv3/Conv2DConv2D.model_13/vgg19/block4_conv2/Relu:activations:09model_13/vgg19/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
2model_13/vgg19/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block4_conv3/BiasAddBiasAdd+model_13/vgg19/block4_conv3/Conv2D:output:0:model_13/vgg19/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 model_13/vgg19/block4_conv3/ReluRelu,model_13/vgg19/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
1model_13/vgg19/block4_conv4/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
"model_13/vgg19/block4_conv4/Conv2DConv2D.model_13/vgg19/block4_conv3/Relu:activations:09model_13/vgg19/block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
2model_13/vgg19/block4_conv4/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block4_conv4/BiasAddBiasAdd+model_13/vgg19/block4_conv4/Conv2D:output:0:model_13/vgg19/block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 model_13/vgg19/block4_conv4/ReluRelu,model_13/vgg19/block4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
"model_13/vgg19/block4_pool/MaxPoolMaxPool.model_13/vgg19/block4_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¶
1model_13/vgg19/block5_conv1/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0÷
"model_13/vgg19/block5_conv1/Conv2DConv2D+model_13/vgg19/block4_pool/MaxPool:output:09model_13/vgg19/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
2model_13/vgg19/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block5_conv1/BiasAddBiasAdd+model_13/vgg19/block5_conv1/Conv2D:output:0:model_13/vgg19/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 model_13/vgg19/block5_conv1/ReluRelu,model_13/vgg19/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
1model_13/vgg19/block5_conv2/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
"model_13/vgg19/block5_conv2/Conv2DConv2D.model_13/vgg19/block5_conv1/Relu:activations:09model_13/vgg19/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
2model_13/vgg19/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block5_conv2/BiasAddBiasAdd+model_13/vgg19/block5_conv2/Conv2D:output:0:model_13/vgg19/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 model_13/vgg19/block5_conv2/ReluRelu,model_13/vgg19/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
1model_13/vgg19/block5_conv3/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
"model_13/vgg19/block5_conv3/Conv2DConv2D.model_13/vgg19/block5_conv2/Relu:activations:09model_13/vgg19/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
2model_13/vgg19/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block5_conv3/BiasAddBiasAdd+model_13/vgg19/block5_conv3/Conv2D:output:0:model_13/vgg19/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 model_13/vgg19/block5_conv3/ReluRelu,model_13/vgg19/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
1model_13/vgg19/block5_conv4/Conv2D/ReadVariableOpReadVariableOp:model_13_vgg19_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
"model_13/vgg19/block5_conv4/Conv2DConv2D.model_13/vgg19/block5_conv3/Relu:activations:09model_13/vgg19/block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
2model_13/vgg19/block5_conv4/BiasAdd/ReadVariableOpReadVariableOp;model_13_vgg19_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ò
#model_13/vgg19/block5_conv4/BiasAddBiasAdd+model_13/vgg19/block5_conv4/Conv2D:output:0:model_13/vgg19/block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 model_13/vgg19/block5_conv4/ReluRelu,model_13/vgg19/block5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
"model_13/vgg19/block5_pool/MaxPoolMaxPool.model_13/vgg19/block5_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

:model_13/global_average_pooling2d_9/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Õ
(model_13/global_average_pooling2d_9/MeanMean+model_13/vgg19/block5_pool/MaxPool:output:0Cmodel_13/global_average_pooling2d_9/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model_13/dense_13/MatMul/ReadVariableOpReadVariableOp0model_13_dense_13_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0¸
model_13/dense_13/MatMulMatMul1model_13/global_average_pooling2d_9/Mean:output:0/model_13/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(model_13/dense_13/BiasAdd/ReadVariableOpReadVariableOp1model_13_dense_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¬
model_13/dense_13/BiasAddBiasAdd"model_13/dense_13/MatMul:product:00model_13/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¶
3model_13/vgg19/block1_conv1/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ù
$model_13/vgg19/block1_conv1/Conv2D_1Conv2Dinputs_1;model_13/vgg19/block1_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
¬
4model_13/vgg19/block1_conv1/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ù
%model_13/vgg19/block1_conv1/BiasAdd_1BiasAdd-model_13/vgg19/block1_conv1/Conv2D_1:output:0<model_13/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
"model_13/vgg19/block1_conv1/Relu_1Relu.model_13/vgg19/block1_conv1/BiasAdd_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@¶
3model_13/vgg19/block1_conv2/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
$model_13/vgg19/block1_conv2/Conv2D_1Conv2D0model_13/vgg19/block1_conv1/Relu_1:activations:0;model_13/vgg19/block1_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
¬
4model_13/vgg19/block1_conv2/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ù
%model_13/vgg19/block1_conv2/BiasAdd_1BiasAdd-model_13/vgg19/block1_conv2/Conv2D_1:output:0<model_13/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
"model_13/vgg19/block1_conv2/Relu_1Relu.model_13/vgg19/block1_conv2/BiasAdd_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@Î
$model_13/vgg19/block1_pool/MaxPool_1MaxPool0model_13/vgg19/block1_conv2/Relu_1:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@*
ksize
*
paddingVALID*
strides
·
3model_13/vgg19/block2_conv1/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ý
$model_13/vgg19/block2_conv1/Conv2D_1Conv2D-model_13/vgg19/block1_pool/MaxPool_1:output:0;model_13/vgg19/block2_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
­
4model_13/vgg19/block2_conv1/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block2_conv1/BiasAdd_1BiasAdd-model_13/vgg19/block2_conv1/Conv2D_1:output:0<model_13/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
"model_13/vgg19/block2_conv1/Relu_1Relu.model_13/vgg19/block2_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd¸
3model_13/vgg19/block2_conv2/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$model_13/vgg19/block2_conv2/Conv2D_1Conv2D0model_13/vgg19/block2_conv1/Relu_1:activations:0;model_13/vgg19/block2_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
­
4model_13/vgg19/block2_conv2/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block2_conv2/BiasAdd_1BiasAdd-model_13/vgg19/block2_conv2/Conv2D_1:output:0<model_13/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
"model_13/vgg19/block2_conv2/Relu_1Relu.model_13/vgg19/block2_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddÏ
$model_13/vgg19/block2_pool/MaxPool_1MaxPool0model_13/vgg19/block2_conv2/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
ksize
*
paddingVALID*
strides
¸
3model_13/vgg19/block3_conv1/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
$model_13/vgg19/block3_conv1/Conv2D_1Conv2D-model_13/vgg19/block2_pool/MaxPool_1:output:0;model_13/vgg19/block3_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
­
4model_13/vgg19/block3_conv1/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block3_conv1/BiasAdd_1BiasAdd-model_13/vgg19/block3_conv1/Conv2D_1:output:0<model_13/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
"model_13/vgg19/block3_conv1/Relu_1Relu.model_13/vgg19/block3_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¸
3model_13/vgg19/block3_conv2/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$model_13/vgg19/block3_conv2/Conv2D_1Conv2D0model_13/vgg19/block3_conv1/Relu_1:activations:0;model_13/vgg19/block3_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
­
4model_13/vgg19/block3_conv2/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block3_conv2/BiasAdd_1BiasAdd-model_13/vgg19/block3_conv2/Conv2D_1:output:0<model_13/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
"model_13/vgg19/block3_conv2/Relu_1Relu.model_13/vgg19/block3_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¸
3model_13/vgg19/block3_conv3/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$model_13/vgg19/block3_conv3/Conv2D_1Conv2D0model_13/vgg19/block3_conv2/Relu_1:activations:0;model_13/vgg19/block3_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
­
4model_13/vgg19/block3_conv3/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block3_conv3/BiasAdd_1BiasAdd-model_13/vgg19/block3_conv3/Conv2D_1:output:0<model_13/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
"model_13/vgg19/block3_conv3/Relu_1Relu.model_13/vgg19/block3_conv3/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¸
3model_13/vgg19/block3_conv4/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$model_13/vgg19/block3_conv4/Conv2D_1Conv2D0model_13/vgg19/block3_conv3/Relu_1:activations:0;model_13/vgg19/block3_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
­
4model_13/vgg19/block3_conv4/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block3_conv4/BiasAdd_1BiasAdd-model_13/vgg19/block3_conv4/Conv2D_1:output:0<model_13/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
"model_13/vgg19/block3_conv4/Relu_1Relu.model_13/vgg19/block3_conv4/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Ï
$model_13/vgg19/block3_pool/MaxPool_1MaxPool0model_13/vgg19/block3_conv4/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¸
3model_13/vgg19/block4_conv1/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
$model_13/vgg19/block4_conv1/Conv2D_1Conv2D-model_13/vgg19/block3_pool/MaxPool_1:output:0;model_13/vgg19/block4_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
­
4model_13/vgg19/block4_conv1/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block4_conv1/BiasAdd_1BiasAdd-model_13/vgg19/block4_conv1/Conv2D_1:output:0<model_13/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"model_13/vgg19/block4_conv1/Relu_1Relu.model_13/vgg19/block4_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
3model_13/vgg19/block4_conv2/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$model_13/vgg19/block4_conv2/Conv2D_1Conv2D0model_13/vgg19/block4_conv1/Relu_1:activations:0;model_13/vgg19/block4_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
­
4model_13/vgg19/block4_conv2/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block4_conv2/BiasAdd_1BiasAdd-model_13/vgg19/block4_conv2/Conv2D_1:output:0<model_13/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"model_13/vgg19/block4_conv2/Relu_1Relu.model_13/vgg19/block4_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
3model_13/vgg19/block4_conv3/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$model_13/vgg19/block4_conv3/Conv2D_1Conv2D0model_13/vgg19/block4_conv2/Relu_1:activations:0;model_13/vgg19/block4_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
­
4model_13/vgg19/block4_conv3/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block4_conv3/BiasAdd_1BiasAdd-model_13/vgg19/block4_conv3/Conv2D_1:output:0<model_13/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"model_13/vgg19/block4_conv3/Relu_1Relu.model_13/vgg19/block4_conv3/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
3model_13/vgg19/block4_conv4/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$model_13/vgg19/block4_conv4/Conv2D_1Conv2D0model_13/vgg19/block4_conv3/Relu_1:activations:0;model_13/vgg19/block4_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
­
4model_13/vgg19/block4_conv4/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block4_conv4/BiasAdd_1BiasAdd-model_13/vgg19/block4_conv4/Conv2D_1:output:0<model_13/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"model_13/vgg19/block4_conv4/Relu_1Relu.model_13/vgg19/block4_conv4/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
$model_13/vgg19/block4_pool/MaxPool_1MaxPool0model_13/vgg19/block4_conv4/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¸
3model_13/vgg19/block5_conv1/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
$model_13/vgg19/block5_conv1/Conv2D_1Conv2D-model_13/vgg19/block4_pool/MaxPool_1:output:0;model_13/vgg19/block5_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
­
4model_13/vgg19/block5_conv1/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block5_conv1/BiasAdd_1BiasAdd-model_13/vgg19/block5_conv1/Conv2D_1:output:0<model_13/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"model_13/vgg19/block5_conv1/Relu_1Relu.model_13/vgg19/block5_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
3model_13/vgg19/block5_conv2/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$model_13/vgg19/block5_conv2/Conv2D_1Conv2D0model_13/vgg19/block5_conv1/Relu_1:activations:0;model_13/vgg19/block5_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
­
4model_13/vgg19/block5_conv2/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block5_conv2/BiasAdd_1BiasAdd-model_13/vgg19/block5_conv2/Conv2D_1:output:0<model_13/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"model_13/vgg19/block5_conv2/Relu_1Relu.model_13/vgg19/block5_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
3model_13/vgg19/block5_conv3/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$model_13/vgg19/block5_conv3/Conv2D_1Conv2D0model_13/vgg19/block5_conv2/Relu_1:activations:0;model_13/vgg19/block5_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
­
4model_13/vgg19/block5_conv3/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block5_conv3/BiasAdd_1BiasAdd-model_13/vgg19/block5_conv3/Conv2D_1:output:0<model_13/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"model_13/vgg19/block5_conv3/Relu_1Relu.model_13/vgg19/block5_conv3/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
3model_13/vgg19/block5_conv4/Conv2D_1/ReadVariableOpReadVariableOp:model_13_vgg19_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$model_13/vgg19/block5_conv4/Conv2D_1Conv2D0model_13/vgg19/block5_conv3/Relu_1:activations:0;model_13/vgg19/block5_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
­
4model_13/vgg19/block5_conv4/BiasAdd_1/ReadVariableOpReadVariableOp;model_13_vgg19_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%model_13/vgg19/block5_conv4/BiasAdd_1BiasAdd-model_13/vgg19/block5_conv4/Conv2D_1:output:0<model_13/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"model_13/vgg19/block5_conv4/Relu_1Relu.model_13/vgg19/block5_conv4/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
$model_13/vgg19/block5_pool/MaxPool_1MaxPool0model_13/vgg19/block5_conv4/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

<model_13/global_average_pooling2d_9/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Û
*model_13/global_average_pooling2d_9/Mean_1Mean-model_13/vgg19/block5_pool/MaxPool_1:output:0Emodel_13/global_average_pooling2d_9/Mean_1/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)model_13/dense_13/MatMul_1/ReadVariableOpReadVariableOp0model_13_dense_13_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0¾
model_13/dense_13/MatMul_1MatMul3model_13/global_average_pooling2d_9/Mean_1:output:01model_13/dense_13/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*model_13/dense_13/BiasAdd_1/ReadVariableOpReadVariableOp1model_13_dense_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0²
model_13/dense_13/BiasAdd_1BiasAdd$model_13/dense_13/MatMul_1:product:02model_13/dense_13/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
lambda_4/subSub"model_13/dense_13/BiasAdd:output:0$model_13/dense_13/BiasAdd_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
lambda_4/SquareSquarelambda_4/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
lambda_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
lambda_4/SumSumlambda_4/Square:y:0'lambda_4/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(W
lambda_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
lambda_4/MaximumMaximumlambda_4/Sum:output:0lambda_4/Maximum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
lambda_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
lambda_4/Maximum_1Maximumlambda_4/Maximum:z:0lambda_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lambda_4/SqrtSqrtlambda_4/Maximum_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_14/MatMulMatMullambda_4/Sqrt:y:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_14/SigmoidSigmoiddense_14/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitydense_14/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp)^model_13/dense_13/BiasAdd/ReadVariableOp+^model_13/dense_13/BiasAdd_1/ReadVariableOp(^model_13/dense_13/MatMul/ReadVariableOp*^model_13/dense_13/MatMul_1/ReadVariableOp3^model_13/vgg19/block1_conv1/BiasAdd/ReadVariableOp5^model_13/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block1_conv1/Conv2D/ReadVariableOp4^model_13/vgg19/block1_conv1/Conv2D_1/ReadVariableOp3^model_13/vgg19/block1_conv2/BiasAdd/ReadVariableOp5^model_13/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block1_conv2/Conv2D/ReadVariableOp4^model_13/vgg19/block1_conv2/Conv2D_1/ReadVariableOp3^model_13/vgg19/block2_conv1/BiasAdd/ReadVariableOp5^model_13/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block2_conv1/Conv2D/ReadVariableOp4^model_13/vgg19/block2_conv1/Conv2D_1/ReadVariableOp3^model_13/vgg19/block2_conv2/BiasAdd/ReadVariableOp5^model_13/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block2_conv2/Conv2D/ReadVariableOp4^model_13/vgg19/block2_conv2/Conv2D_1/ReadVariableOp3^model_13/vgg19/block3_conv1/BiasAdd/ReadVariableOp5^model_13/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block3_conv1/Conv2D/ReadVariableOp4^model_13/vgg19/block3_conv1/Conv2D_1/ReadVariableOp3^model_13/vgg19/block3_conv2/BiasAdd/ReadVariableOp5^model_13/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block3_conv2/Conv2D/ReadVariableOp4^model_13/vgg19/block3_conv2/Conv2D_1/ReadVariableOp3^model_13/vgg19/block3_conv3/BiasAdd/ReadVariableOp5^model_13/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block3_conv3/Conv2D/ReadVariableOp4^model_13/vgg19/block3_conv3/Conv2D_1/ReadVariableOp3^model_13/vgg19/block3_conv4/BiasAdd/ReadVariableOp5^model_13/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block3_conv4/Conv2D/ReadVariableOp4^model_13/vgg19/block3_conv4/Conv2D_1/ReadVariableOp3^model_13/vgg19/block4_conv1/BiasAdd/ReadVariableOp5^model_13/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block4_conv1/Conv2D/ReadVariableOp4^model_13/vgg19/block4_conv1/Conv2D_1/ReadVariableOp3^model_13/vgg19/block4_conv2/BiasAdd/ReadVariableOp5^model_13/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block4_conv2/Conv2D/ReadVariableOp4^model_13/vgg19/block4_conv2/Conv2D_1/ReadVariableOp3^model_13/vgg19/block4_conv3/BiasAdd/ReadVariableOp5^model_13/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block4_conv3/Conv2D/ReadVariableOp4^model_13/vgg19/block4_conv3/Conv2D_1/ReadVariableOp3^model_13/vgg19/block4_conv4/BiasAdd/ReadVariableOp5^model_13/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block4_conv4/Conv2D/ReadVariableOp4^model_13/vgg19/block4_conv4/Conv2D_1/ReadVariableOp3^model_13/vgg19/block5_conv1/BiasAdd/ReadVariableOp5^model_13/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block5_conv1/Conv2D/ReadVariableOp4^model_13/vgg19/block5_conv1/Conv2D_1/ReadVariableOp3^model_13/vgg19/block5_conv2/BiasAdd/ReadVariableOp5^model_13/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block5_conv2/Conv2D/ReadVariableOp4^model_13/vgg19/block5_conv2/Conv2D_1/ReadVariableOp3^model_13/vgg19/block5_conv3/BiasAdd/ReadVariableOp5^model_13/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block5_conv3/Conv2D/ReadVariableOp4^model_13/vgg19/block5_conv3/Conv2D_1/ReadVariableOp3^model_13/vgg19/block5_conv4/BiasAdd/ReadVariableOp5^model_13/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp2^model_13/vgg19/block5_conv4/Conv2D/ReadVariableOp4^model_13/vgg19/block5_conv4/Conv2D_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2T
(model_13/dense_13/BiasAdd/ReadVariableOp(model_13/dense_13/BiasAdd/ReadVariableOp2X
*model_13/dense_13/BiasAdd_1/ReadVariableOp*model_13/dense_13/BiasAdd_1/ReadVariableOp2R
'model_13/dense_13/MatMul/ReadVariableOp'model_13/dense_13/MatMul/ReadVariableOp2V
)model_13/dense_13/MatMul_1/ReadVariableOp)model_13/dense_13/MatMul_1/ReadVariableOp2h
2model_13/vgg19/block1_conv1/BiasAdd/ReadVariableOp2model_13/vgg19/block1_conv1/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp4model_13/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block1_conv1/Conv2D/ReadVariableOp1model_13/vgg19/block1_conv1/Conv2D/ReadVariableOp2j
3model_13/vgg19/block1_conv1/Conv2D_1/ReadVariableOp3model_13/vgg19/block1_conv1/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block1_conv2/BiasAdd/ReadVariableOp2model_13/vgg19/block1_conv2/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp4model_13/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block1_conv2/Conv2D/ReadVariableOp1model_13/vgg19/block1_conv2/Conv2D/ReadVariableOp2j
3model_13/vgg19/block1_conv2/Conv2D_1/ReadVariableOp3model_13/vgg19/block1_conv2/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block2_conv1/BiasAdd/ReadVariableOp2model_13/vgg19/block2_conv1/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp4model_13/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block2_conv1/Conv2D/ReadVariableOp1model_13/vgg19/block2_conv1/Conv2D/ReadVariableOp2j
3model_13/vgg19/block2_conv1/Conv2D_1/ReadVariableOp3model_13/vgg19/block2_conv1/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block2_conv2/BiasAdd/ReadVariableOp2model_13/vgg19/block2_conv2/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp4model_13/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block2_conv2/Conv2D/ReadVariableOp1model_13/vgg19/block2_conv2/Conv2D/ReadVariableOp2j
3model_13/vgg19/block2_conv2/Conv2D_1/ReadVariableOp3model_13/vgg19/block2_conv2/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block3_conv1/BiasAdd/ReadVariableOp2model_13/vgg19/block3_conv1/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp4model_13/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block3_conv1/Conv2D/ReadVariableOp1model_13/vgg19/block3_conv1/Conv2D/ReadVariableOp2j
3model_13/vgg19/block3_conv1/Conv2D_1/ReadVariableOp3model_13/vgg19/block3_conv1/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block3_conv2/BiasAdd/ReadVariableOp2model_13/vgg19/block3_conv2/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp4model_13/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block3_conv2/Conv2D/ReadVariableOp1model_13/vgg19/block3_conv2/Conv2D/ReadVariableOp2j
3model_13/vgg19/block3_conv2/Conv2D_1/ReadVariableOp3model_13/vgg19/block3_conv2/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block3_conv3/BiasAdd/ReadVariableOp2model_13/vgg19/block3_conv3/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp4model_13/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block3_conv3/Conv2D/ReadVariableOp1model_13/vgg19/block3_conv3/Conv2D/ReadVariableOp2j
3model_13/vgg19/block3_conv3/Conv2D_1/ReadVariableOp3model_13/vgg19/block3_conv3/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block3_conv4/BiasAdd/ReadVariableOp2model_13/vgg19/block3_conv4/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp4model_13/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block3_conv4/Conv2D/ReadVariableOp1model_13/vgg19/block3_conv4/Conv2D/ReadVariableOp2j
3model_13/vgg19/block3_conv4/Conv2D_1/ReadVariableOp3model_13/vgg19/block3_conv4/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block4_conv1/BiasAdd/ReadVariableOp2model_13/vgg19/block4_conv1/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp4model_13/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block4_conv1/Conv2D/ReadVariableOp1model_13/vgg19/block4_conv1/Conv2D/ReadVariableOp2j
3model_13/vgg19/block4_conv1/Conv2D_1/ReadVariableOp3model_13/vgg19/block4_conv1/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block4_conv2/BiasAdd/ReadVariableOp2model_13/vgg19/block4_conv2/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp4model_13/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block4_conv2/Conv2D/ReadVariableOp1model_13/vgg19/block4_conv2/Conv2D/ReadVariableOp2j
3model_13/vgg19/block4_conv2/Conv2D_1/ReadVariableOp3model_13/vgg19/block4_conv2/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block4_conv3/BiasAdd/ReadVariableOp2model_13/vgg19/block4_conv3/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp4model_13/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block4_conv3/Conv2D/ReadVariableOp1model_13/vgg19/block4_conv3/Conv2D/ReadVariableOp2j
3model_13/vgg19/block4_conv3/Conv2D_1/ReadVariableOp3model_13/vgg19/block4_conv3/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block4_conv4/BiasAdd/ReadVariableOp2model_13/vgg19/block4_conv4/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp4model_13/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block4_conv4/Conv2D/ReadVariableOp1model_13/vgg19/block4_conv4/Conv2D/ReadVariableOp2j
3model_13/vgg19/block4_conv4/Conv2D_1/ReadVariableOp3model_13/vgg19/block4_conv4/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block5_conv1/BiasAdd/ReadVariableOp2model_13/vgg19/block5_conv1/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp4model_13/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block5_conv1/Conv2D/ReadVariableOp1model_13/vgg19/block5_conv1/Conv2D/ReadVariableOp2j
3model_13/vgg19/block5_conv1/Conv2D_1/ReadVariableOp3model_13/vgg19/block5_conv1/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block5_conv2/BiasAdd/ReadVariableOp2model_13/vgg19/block5_conv2/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp4model_13/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block5_conv2/Conv2D/ReadVariableOp1model_13/vgg19/block5_conv2/Conv2D/ReadVariableOp2j
3model_13/vgg19/block5_conv2/Conv2D_1/ReadVariableOp3model_13/vgg19/block5_conv2/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block5_conv3/BiasAdd/ReadVariableOp2model_13/vgg19/block5_conv3/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp4model_13/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block5_conv3/Conv2D/ReadVariableOp1model_13/vgg19/block5_conv3/Conv2D/ReadVariableOp2j
3model_13/vgg19/block5_conv3/Conv2D_1/ReadVariableOp3model_13/vgg19/block5_conv3/Conv2D_1/ReadVariableOp2h
2model_13/vgg19/block5_conv4/BiasAdd/ReadVariableOp2model_13/vgg19/block5_conv4/BiasAdd/ReadVariableOp2l
4model_13/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp4model_13/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp2f
1model_13/vgg19/block5_conv4/Conv2D/ReadVariableOp1model_13/vgg19/block5_conv4/Conv2D/ReadVariableOp2j
3model_13/vgg19/block5_conv4/Conv2D_1/ReadVariableOp3model_13/vgg19/block5_conv4/Conv2D_1/ReadVariableOp:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
inputs/1


H__inference_block5_conv4_layer_call_and_return_conditional_losses_975093

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block5_conv4_layer_call_and_return_conditional_losses_979216

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
H
,__inference_block5_pool_layer_call_fn_979221

inputs
identityØ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_pool_layer_call_and_return_conditional_losses_974813
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
¥
-__inference_block5_conv4_layer_call_fn_979205

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv4_layer_call_and_return_conditional_losses_975093x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

c
G__inference_block4_pool_layer_call_and_return_conditional_losses_979136

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô
Ò

D__inference_model_13_layer_call_and_return_conditional_losses_975905

inputs&
vgg19_975822:@
vgg19_975824:@&
vgg19_975826:@@
vgg19_975828:@'
vgg19_975830:@
vgg19_975832:	(
vgg19_975834:
vgg19_975836:	(
vgg19_975838:
vgg19_975840:	(
vgg19_975842:
vgg19_975844:	(
vgg19_975846:
vgg19_975848:	(
vgg19_975850:
vgg19_975852:	(
vgg19_975854:
vgg19_975856:	(
vgg19_975858:
vgg19_975860:	(
vgg19_975862:
vgg19_975864:	(
vgg19_975866:
vgg19_975868:	(
vgg19_975870:
vgg19_975872:	(
vgg19_975874:
vgg19_975876:	(
vgg19_975878:
vgg19_975880:	(
vgg19_975882:
vgg19_975884:	"
dense_13_975899:	@
dense_13_975901:@
identity¢ dense_13/StatefulPartitionedCall¢vgg19/StatefulPartitionedCallÐ
vgg19/StatefulPartitionedCallStatefulPartitionedCallinputsvgg19_975822vgg19_975824vgg19_975826vgg19_975828vgg19_975830vgg19_975832vgg19_975834vgg19_975836vgg19_975838vgg19_975840vgg19_975842vgg19_975844vgg19_975846vgg19_975848vgg19_975850vgg19_975852vgg19_975854vgg19_975856vgg19_975858vgg19_975860vgg19_975862vgg19_975864vgg19_975866vgg19_975868vgg19_975870vgg19_975872vgg19_975874vgg19_975876vgg19_975878vgg19_975880vgg19_975882vgg19_975884*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_vgg19_layer_call_and_return_conditional_losses_975101
*global_average_pooling2d_9/PartitionedCallPartitionedCall&vgg19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *_
fZRX
V__inference_global_average_pooling2d_9_layer_call_and_return_conditional_losses_975812 
 dense_13/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_9/PartitionedCall:output:0dense_13_975899dense_13_975901*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_975898x
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp!^dense_13/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2>
vgg19/StatefulPartitionedCallvgg19/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs
©g

A__inference_vgg19_layer_call_and_return_conditional_losses_975802
input_30-
block1_conv1_975716:@!
block1_conv1_975718:@-
block1_conv2_975721:@@!
block1_conv2_975723:@.
block2_conv1_975727:@"
block2_conv1_975729:	/
block2_conv2_975732:"
block2_conv2_975734:	/
block3_conv1_975738:"
block3_conv1_975740:	/
block3_conv2_975743:"
block3_conv2_975745:	/
block3_conv3_975748:"
block3_conv3_975750:	/
block3_conv4_975753:"
block3_conv4_975755:	/
block4_conv1_975759:"
block4_conv1_975761:	/
block4_conv2_975764:"
block4_conv2_975766:	/
block4_conv3_975769:"
block4_conv3_975771:	/
block4_conv4_975774:"
block4_conv4_975776:	/
block5_conv1_975780:"
block5_conv1_975782:	/
block5_conv2_975785:"
block5_conv2_975787:	/
block5_conv3_975790:"
block5_conv3_975792:	/
block5_conv4_975795:"
block5_conv4_975797:	
identity¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block3_conv4/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall¢$block4_conv2/StatefulPartitionedCall¢$block4_conv3/StatefulPartitionedCall¢$block4_conv4/StatefulPartitionedCall¢$block5_conv1/StatefulPartitionedCall¢$block5_conv2/StatefulPartitionedCall¢$block5_conv3/StatefulPartitionedCall¢$block5_conv4/StatefulPartitionedCall
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_30block1_conv1_975716block1_conv1_975718*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block1_conv1_layer_call_and_return_conditional_losses_974834´
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_975721block1_conv2_975723*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block1_conv2_layer_call_and_return_conditional_losses_974851ð
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block1_pool_layer_call_and_return_conditional_losses_974765ª
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_975727block2_conv1_975729*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block2_conv1_layer_call_and_return_conditional_losses_974869³
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_975732block2_conv2_975734*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block2_conv2_layer_call_and_return_conditional_losses_974886ñ
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block2_pool_layer_call_and_return_conditional_losses_974777ª
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_975738block3_conv1_975740*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv1_layer_call_and_return_conditional_losses_974904³
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_975743block3_conv2_975745*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv2_layer_call_and_return_conditional_losses_974921³
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_975748block3_conv3_975750*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv3_layer_call_and_return_conditional_losses_974938³
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_975753block3_conv4_975755*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv4_layer_call_and_return_conditional_losses_974955ñ
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_pool_layer_call_and_return_conditional_losses_974789ª
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_975759block4_conv1_975761*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv1_layer_call_and_return_conditional_losses_974973³
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_975764block4_conv2_975766*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv2_layer_call_and_return_conditional_losses_974990³
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_975769block4_conv3_975771*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv3_layer_call_and_return_conditional_losses_975007³
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_975774block4_conv4_975776*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv4_layer_call_and_return_conditional_losses_975024ñ
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_pool_layer_call_and_return_conditional_losses_974801ª
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_975780block5_conv1_975782*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv1_layer_call_and_return_conditional_losses_975042³
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_975785block5_conv2_975787*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv2_layer_call_and_return_conditional_losses_975059³
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_975790block5_conv3_975792*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv3_layer_call_and_return_conditional_losses_975076³
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_975795block5_conv4_975797*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv4_layer_call_and_return_conditional_losses_975093ñ
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_pool_layer_call_and_return_conditional_losses_974813|
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
input_30
ü
¥
-__inference_block4_conv2_layer_call_fn_979075

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv2_layer_call_and_return_conditional_losses_974990x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢)
¬
D__inference_model_14_layer_call_and_return_conditional_losses_977153
input_27
input_28)
model_13_977042:@
model_13_977044:@)
model_13_977046:@@
model_13_977048:@*
model_13_977050:@
model_13_977052:	+
model_13_977054:
model_13_977056:	+
model_13_977058:
model_13_977060:	+
model_13_977062:
model_13_977064:	+
model_13_977066:
model_13_977068:	+
model_13_977070:
model_13_977072:	+
model_13_977074:
model_13_977076:	+
model_13_977078:
model_13_977080:	+
model_13_977082:
model_13_977084:	+
model_13_977086:
model_13_977088:	+
model_13_977090:
model_13_977092:	+
model_13_977094:
model_13_977096:	+
model_13_977098:
model_13_977100:	+
model_13_977102:
model_13_977104:	"
model_13_977106:	@
model_13_977108:@!
dense_14_977147:
dense_14_977149:
identity¢ dense_14/StatefulPartitionedCall¢ model_13/StatefulPartitionedCall¢"model_13/StatefulPartitionedCall_1Õ
 model_13/StatefulPartitionedCallStatefulPartitionedCallinput_27model_13_977042model_13_977044model_13_977046model_13_977048model_13_977050model_13_977052model_13_977054model_13_977056model_13_977058model_13_977060model_13_977062model_13_977064model_13_977066model_13_977068model_13_977070model_13_977072model_13_977074model_13_977076model_13_977078model_13_977080model_13_977082model_13_977084model_13_977086model_13_977088model_13_977090model_13_977092model_13_977094model_13_977096model_13_977098model_13_977100model_13_977102model_13_977104model_13_977106model_13_977108*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_13_layer_call_and_return_conditional_losses_975905×
"model_13/StatefulPartitionedCall_1StatefulPartitionedCallinput_28model_13_977042model_13_977044model_13_977046model_13_977048model_13_977050model_13_977052model_13_977054model_13_977056model_13_977058model_13_977060model_13_977062model_13_977064model_13_977066model_13_977068model_13_977070model_13_977072model_13_977074model_13_977076model_13_977078model_13_977080model_13_977082model_13_977084model_13_977086model_13_977088model_13_977090model_13_977092model_13_977094model_13_977096model_13_977098model_13_977100model_13_977102model_13_977104model_13_977106model_13_977108*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_13_layer_call_and_return_conditional_losses_975905
lambda_4/PartitionedCallPartitionedCall)model_13/StatefulPartitionedCall:output:0+model_13/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_lambda_4_layer_call_and_return_conditional_losses_976557
 dense_14/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0dense_14_977147dense_14_977149*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_976570x
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
NoOpNoOp!^dense_14/StatefulPartitionedCall!^model_13/StatefulPartitionedCall#^model_13/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 model_13/StatefulPartitionedCall model_13/StatefulPartitionedCall2H
"model_13/StatefulPartitionedCall_1"model_13/StatefulPartitionedCall_1:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
input_27:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
input_28


H__inference_block3_conv1_layer_call_and_return_conditional_losses_978976

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs


p
D__inference_lambda_4_layer_call_and_return_conditional_losses_978426
inputs_0
inputs_1
identityP
subSubinputs_0inputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@K
SquareSquaresub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :y
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3f
MaximumMaximumSum:output:0Maximum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    c
	Maximum_1MaximumMaximum:z:0Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
SqrtSqrtMaximum_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/1
£g

A__inference_vgg19_layer_call_and_return_conditional_losses_975488

inputs-
block1_conv1_975402:@!
block1_conv1_975404:@-
block1_conv2_975407:@@!
block1_conv2_975409:@.
block2_conv1_975413:@"
block2_conv1_975415:	/
block2_conv2_975418:"
block2_conv2_975420:	/
block3_conv1_975424:"
block3_conv1_975426:	/
block3_conv2_975429:"
block3_conv2_975431:	/
block3_conv3_975434:"
block3_conv3_975436:	/
block3_conv4_975439:"
block3_conv4_975441:	/
block4_conv1_975445:"
block4_conv1_975447:	/
block4_conv2_975450:"
block4_conv2_975452:	/
block4_conv3_975455:"
block4_conv3_975457:	/
block4_conv4_975460:"
block4_conv4_975462:	/
block5_conv1_975466:"
block5_conv1_975468:	/
block5_conv2_975471:"
block5_conv2_975473:	/
block5_conv3_975476:"
block5_conv3_975478:	/
block5_conv4_975481:"
block5_conv4_975483:	
identity¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block3_conv4/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall¢$block4_conv2/StatefulPartitionedCall¢$block4_conv3/StatefulPartitionedCall¢$block4_conv4/StatefulPartitionedCall¢$block5_conv1/StatefulPartitionedCall¢$block5_conv2/StatefulPartitionedCall¢$block5_conv3/StatefulPartitionedCall¢$block5_conv4/StatefulPartitionedCall
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_975402block1_conv1_975404*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block1_conv1_layer_call_and_return_conditional_losses_974834´
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_975407block1_conv2_975409*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block1_conv2_layer_call_and_return_conditional_losses_974851ð
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block1_pool_layer_call_and_return_conditional_losses_974765ª
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_975413block2_conv1_975415*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block2_conv1_layer_call_and_return_conditional_losses_974869³
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_975418block2_conv2_975420*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block2_conv2_layer_call_and_return_conditional_losses_974886ñ
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block2_pool_layer_call_and_return_conditional_losses_974777ª
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_975424block3_conv1_975426*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv1_layer_call_and_return_conditional_losses_974904³
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_975429block3_conv2_975431*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv2_layer_call_and_return_conditional_losses_974921³
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_975434block3_conv3_975436*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv3_layer_call_and_return_conditional_losses_974938³
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_975439block3_conv4_975441*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv4_layer_call_and_return_conditional_losses_974955ñ
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_pool_layer_call_and_return_conditional_losses_974789ª
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_975445block4_conv1_975447*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv1_layer_call_and_return_conditional_losses_974973³
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_975450block4_conv2_975452*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv2_layer_call_and_return_conditional_losses_974990³
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_975455block4_conv3_975457*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv3_layer_call_and_return_conditional_losses_975007³
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_975460block4_conv4_975462*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv4_layer_call_and_return_conditional_losses_975024ñ
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_pool_layer_call_and_return_conditional_losses_974801ª
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_975466block5_conv1_975468*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv1_layer_call_and_return_conditional_losses_975042³
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_975471block5_conv2_975473*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv2_layer_call_and_return_conditional_losses_975059³
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_975476block5_conv3_975478*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv3_layer_call_and_return_conditional_losses_975076³
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_975481block5_conv4_975483*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv4_layer_call_and_return_conditional_losses_975093ñ
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_pool_layer_call_and_return_conditional_losses_974813|
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs
)
ª
D__inference_model_14_layer_call_and_return_conditional_losses_976577

inputs
inputs_1)
model_13_976439:@
model_13_976441:@)
model_13_976443:@@
model_13_976445:@*
model_13_976447:@
model_13_976449:	+
model_13_976451:
model_13_976453:	+
model_13_976455:
model_13_976457:	+
model_13_976459:
model_13_976461:	+
model_13_976463:
model_13_976465:	+
model_13_976467:
model_13_976469:	+
model_13_976471:
model_13_976473:	+
model_13_976475:
model_13_976477:	+
model_13_976479:
model_13_976481:	+
model_13_976483:
model_13_976485:	+
model_13_976487:
model_13_976489:	+
model_13_976491:
model_13_976493:	+
model_13_976495:
model_13_976497:	+
model_13_976499:
model_13_976501:	"
model_13_976503:	@
model_13_976505:@!
dense_14_976571:
dense_14_976573:
identity¢ dense_14/StatefulPartitionedCall¢ model_13/StatefulPartitionedCall¢"model_13/StatefulPartitionedCall_1Ó
 model_13/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_13_976439model_13_976441model_13_976443model_13_976445model_13_976447model_13_976449model_13_976451model_13_976453model_13_976455model_13_976457model_13_976459model_13_976461model_13_976463model_13_976465model_13_976467model_13_976469model_13_976471model_13_976473model_13_976475model_13_976477model_13_976479model_13_976481model_13_976483model_13_976485model_13_976487model_13_976489model_13_976491model_13_976493model_13_976495model_13_976497model_13_976499model_13_976501model_13_976503model_13_976505*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_13_layer_call_and_return_conditional_losses_975905×
"model_13/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1model_13_976439model_13_976441model_13_976443model_13_976445model_13_976447model_13_976449model_13_976451model_13_976453model_13_976455model_13_976457model_13_976459model_13_976461model_13_976463model_13_976465model_13_976467model_13_976469model_13_976471model_13_976473model_13_976475model_13_976477model_13_976479model_13_976481model_13_976483model_13_976485model_13_976487model_13_976489model_13_976491model_13_976493model_13_976495model_13_976497model_13_976499model_13_976501model_13_976503model_13_976505*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_13_layer_call_and_return_conditional_losses_975905
lambda_4/PartitionedCallPartitionedCall)model_13/StatefulPartitionedCall:output:0+model_13/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_lambda_4_layer_call_and_return_conditional_losses_976557
 dense_14/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0dense_14_976571dense_14_976573*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_976570x
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
NoOpNoOp!^dense_14/StatefulPartitionedCall!^model_13/StatefulPartitionedCall#^model_13/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 model_13/StatefulPartitionedCall model_13/StatefulPartitionedCall2H
"model_13/StatefulPartitionedCall_1"model_13/StatefulPartitionedCall_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs:YU
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs

c
G__inference_block4_pool_layer_call_and_return_conditional_losses_974801

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½
ú	
$__inference_signature_wrapper_977982
input_27
input_28!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	

unknown_31:	@

unknown_32:@

unknown_33:

unknown_34:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_27input_28unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_974756o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
input_27:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
input_28


H__inference_block5_conv1_layer_call_and_return_conditional_losses_975042

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

c
G__inference_block1_pool_layer_call_and_return_conditional_losses_978906

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


n
D__inference_lambda_4_layer_call_and_return_conditional_losses_976557

inputs
inputs_1
identityN
subSubinputsinputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@K
SquareSquaresub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :y
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3f
MaximumMaximumSum:output:0Maximum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    c
	Maximum_1MaximumMaximum:z:0Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
SqrtSqrtMaximum_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

W
;__inference_global_average_pooling2d_9_layer_call_fn_978831

inputs
identityÍ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *_
fZRX
V__inference_global_average_pooling2d_9_layer_call_and_return_conditional_losses_975812i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú
Ô

D__inference_model_13_layer_call_and_return_conditional_losses_976430
input_29&
vgg19_976358:@
vgg19_976360:@&
vgg19_976362:@@
vgg19_976364:@'
vgg19_976366:@
vgg19_976368:	(
vgg19_976370:
vgg19_976372:	(
vgg19_976374:
vgg19_976376:	(
vgg19_976378:
vgg19_976380:	(
vgg19_976382:
vgg19_976384:	(
vgg19_976386:
vgg19_976388:	(
vgg19_976390:
vgg19_976392:	(
vgg19_976394:
vgg19_976396:	(
vgg19_976398:
vgg19_976400:	(
vgg19_976402:
vgg19_976404:	(
vgg19_976406:
vgg19_976408:	(
vgg19_976410:
vgg19_976412:	(
vgg19_976414:
vgg19_976416:	(
vgg19_976418:
vgg19_976420:	"
dense_13_976424:	@
dense_13_976426:@
identity¢ dense_13/StatefulPartitionedCall¢vgg19/StatefulPartitionedCallÒ
vgg19/StatefulPartitionedCallStatefulPartitionedCallinput_29vgg19_976358vgg19_976360vgg19_976362vgg19_976364vgg19_976366vgg19_976368vgg19_976370vgg19_976372vgg19_976374vgg19_976376vgg19_976378vgg19_976380vgg19_976382vgg19_976384vgg19_976386vgg19_976388vgg19_976390vgg19_976392vgg19_976394vgg19_976396vgg19_976398vgg19_976400vgg19_976402vgg19_976404vgg19_976406vgg19_976408vgg19_976410vgg19_976412vgg19_976414vgg19_976416vgg19_976418vgg19_976420*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_vgg19_layer_call_and_return_conditional_losses_975488
*global_average_pooling2d_9/PartitionedCallPartitionedCall&vgg19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *_
fZRX
V__inference_global_average_pooling2d_9_layer_call_and_return_conditional_losses_975812 
 dense_13/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_9/PartitionedCall:output:0dense_13_976424dense_13_976426*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_975898x
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp!^dense_13/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2>
vgg19/StatefulPartitionedCallvgg19/StatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
input_29


H__inference_block3_conv4_layer_call_and_return_conditional_losses_979036

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
ñ
·	
)__inference_model_13_layer_call_fn_978055

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	

unknown_31:	@

unknown_32:@
identity¢StatefulPartitionedCall
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
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_13_layer_call_and_return_conditional_losses_975905o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs


H__inference_block5_conv3_layer_call_and_return_conditional_losses_975076

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
¹	
)__inference_model_13_layer_call_fn_976280
input_29!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	

unknown_31:	@

unknown_32:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_29unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_13_layer_call_and_return_conditional_losses_976136o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
input_29
¥
U
)__inference_lambda_4_layer_call_fn_978398
inputs_0
inputs_1
identity¿
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_lambda_4_layer_call_and_return_conditional_losses_976686`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/1
ü
¥
-__inference_block3_conv1_layer_call_fn_978965

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv1_layer_call_and_return_conditional_losses_974904x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ22: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs

c
G__inference_block5_pool_layer_call_and_return_conditional_losses_979226

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
¹	
)__inference_model_13_layer_call_fn_975976
input_29!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	

unknown_31:	@

unknown_32:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_29unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_13_layer_call_and_return_conditional_losses_975905o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
input_29


H__inference_block5_conv1_layer_call_and_return_conditional_losses_979156

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

)__inference_dense_13_layer_call_fn_978846

inputs
unknown:	@
	unknown_0:@
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_975898o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block5_conv3_layer_call_and_return_conditional_losses_979196

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block4_conv3_layer_call_and_return_conditional_losses_979106

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block3_conv2_layer_call_and_return_conditional_losses_978996

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
ü
¥
-__inference_block2_conv2_layer_call_fn_978935

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block2_conv2_layer_call_and_return_conditional_losses_974886x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿdd: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
ü
¥
-__inference_block3_conv4_layer_call_fn_979025

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv4_layer_call_and_return_conditional_losses_974955x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ22: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs

c
G__inference_block1_pool_layer_call_and_return_conditional_losses_974765

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
¥
-__inference_block4_conv3_layer_call_fn_979095

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv3_layer_call_and_return_conditional_losses_975007x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤
û
&__inference_vgg19_layer_call_fn_978584

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	
identity¢StatefulPartitionedCallý
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
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_vgg19_layer_call_and_return_conditional_losses_975488x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs
©g

A__inference_vgg19_layer_call_and_return_conditional_losses_975713
input_30-
block1_conv1_975627:@!
block1_conv1_975629:@-
block1_conv2_975632:@@!
block1_conv2_975634:@.
block2_conv1_975638:@"
block2_conv1_975640:	/
block2_conv2_975643:"
block2_conv2_975645:	/
block3_conv1_975649:"
block3_conv1_975651:	/
block3_conv2_975654:"
block3_conv2_975656:	/
block3_conv3_975659:"
block3_conv3_975661:	/
block3_conv4_975664:"
block3_conv4_975666:	/
block4_conv1_975670:"
block4_conv1_975672:	/
block4_conv2_975675:"
block4_conv2_975677:	/
block4_conv3_975680:"
block4_conv3_975682:	/
block4_conv4_975685:"
block4_conv4_975687:	/
block5_conv1_975691:"
block5_conv1_975693:	/
block5_conv2_975696:"
block5_conv2_975698:	/
block5_conv3_975701:"
block5_conv3_975703:	/
block5_conv4_975706:"
block5_conv4_975708:	
identity¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block3_conv4/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall¢$block4_conv2/StatefulPartitionedCall¢$block4_conv3/StatefulPartitionedCall¢$block4_conv4/StatefulPartitionedCall¢$block5_conv1/StatefulPartitionedCall¢$block5_conv2/StatefulPartitionedCall¢$block5_conv3/StatefulPartitionedCall¢$block5_conv4/StatefulPartitionedCall
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_30block1_conv1_975627block1_conv1_975629*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block1_conv1_layer_call_and_return_conditional_losses_974834´
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_975632block1_conv2_975634*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block1_conv2_layer_call_and_return_conditional_losses_974851ð
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block1_pool_layer_call_and_return_conditional_losses_974765ª
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_975638block2_conv1_975640*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block2_conv1_layer_call_and_return_conditional_losses_974869³
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_975643block2_conv2_975645*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block2_conv2_layer_call_and_return_conditional_losses_974886ñ
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block2_pool_layer_call_and_return_conditional_losses_974777ª
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_975649block3_conv1_975651*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv1_layer_call_and_return_conditional_losses_974904³
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_975654block3_conv2_975656*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv2_layer_call_and_return_conditional_losses_974921³
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_975659block3_conv3_975661*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv3_layer_call_and_return_conditional_losses_974938³
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_975664block3_conv4_975666*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv4_layer_call_and_return_conditional_losses_974955ñ
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_pool_layer_call_and_return_conditional_losses_974789ª
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_975670block4_conv1_975672*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv1_layer_call_and_return_conditional_losses_974973³
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_975675block4_conv2_975677*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv2_layer_call_and_return_conditional_losses_974990³
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_975680block4_conv3_975682*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv3_layer_call_and_return_conditional_losses_975007³
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_975685block4_conv4_975687*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv4_layer_call_and_return_conditional_losses_975024ñ
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_pool_layer_call_and_return_conditional_losses_974801ª
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_975691block5_conv1_975693*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv1_layer_call_and_return_conditional_losses_975042³
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_975696block5_conv2_975698*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv2_layer_call_and_return_conditional_losses_975059³
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_975701block5_conv3_975703*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv3_layer_call_and_return_conditional_losses_975076³
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_975706block5_conv4_975708*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv4_layer_call_and_return_conditional_losses_975093ñ
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_pool_layer_call_and_return_conditional_losses_974813|
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
input_30
¤
é
A__inference_vgg19_layer_call_and_return_conditional_losses_978705

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@;
,block2_conv1_biasadd_readvariableop_resource:	G
+block2_conv2_conv2d_readvariableop_resource:;
,block2_conv2_biasadd_readvariableop_resource:	G
+block3_conv1_conv2d_readvariableop_resource:;
,block3_conv1_biasadd_readvariableop_resource:	G
+block3_conv2_conv2d_readvariableop_resource:;
,block3_conv2_biasadd_readvariableop_resource:	G
+block3_conv3_conv2d_readvariableop_resource:;
,block3_conv3_biasadd_readvariableop_resource:	G
+block3_conv4_conv2d_readvariableop_resource:;
,block3_conv4_biasadd_readvariableop_resource:	G
+block4_conv1_conv2d_readvariableop_resource:;
,block4_conv1_biasadd_readvariableop_resource:	G
+block4_conv2_conv2d_readvariableop_resource:;
,block4_conv2_biasadd_readvariableop_resource:	G
+block4_conv3_conv2d_readvariableop_resource:;
,block4_conv3_biasadd_readvariableop_resource:	G
+block4_conv4_conv2d_readvariableop_resource:;
,block4_conv4_biasadd_readvariableop_resource:	G
+block5_conv1_conv2d_readvariableop_resource:;
,block5_conv1_biasadd_readvariableop_resource:	G
+block5_conv2_conv2d_readvariableop_resource:;
,block5_conv2_biasadd_readvariableop_resource:	G
+block5_conv3_conv2d_readvariableop_resource:;
,block5_conv3_biasadd_readvariableop_resource:	G
+block5_conv4_conv2d_readvariableop_resource:;
,block5_conv4_biasadd_readvariableop_resource:	
identity¢#block1_conv1/BiasAdd/ReadVariableOp¢"block1_conv1/Conv2D/ReadVariableOp¢#block1_conv2/BiasAdd/ReadVariableOp¢"block1_conv2/Conv2D/ReadVariableOp¢#block2_conv1/BiasAdd/ReadVariableOp¢"block2_conv1/Conv2D/ReadVariableOp¢#block2_conv2/BiasAdd/ReadVariableOp¢"block2_conv2/Conv2D/ReadVariableOp¢#block3_conv1/BiasAdd/ReadVariableOp¢"block3_conv1/Conv2D/ReadVariableOp¢#block3_conv2/BiasAdd/ReadVariableOp¢"block3_conv2/Conv2D/ReadVariableOp¢#block3_conv3/BiasAdd/ReadVariableOp¢"block3_conv3/Conv2D/ReadVariableOp¢#block3_conv4/BiasAdd/ReadVariableOp¢"block3_conv4/Conv2D/ReadVariableOp¢#block4_conv1/BiasAdd/ReadVariableOp¢"block4_conv1/Conv2D/ReadVariableOp¢#block4_conv2/BiasAdd/ReadVariableOp¢"block4_conv2/Conv2D/ReadVariableOp¢#block4_conv3/BiasAdd/ReadVariableOp¢"block4_conv3/Conv2D/ReadVariableOp¢#block4_conv4/BiasAdd/ReadVariableOp¢"block4_conv4/Conv2D/ReadVariableOp¢#block5_conv1/BiasAdd/ReadVariableOp¢"block5_conv1/Conv2D/ReadVariableOp¢#block5_conv2/BiasAdd/ReadVariableOp¢"block5_conv2/Conv2D/ReadVariableOp¢#block5_conv3/BiasAdd/ReadVariableOp¢"block5_conv3/Conv2D/ReadVariableOp¢#block5_conv4/BiasAdd/ReadVariableOp¢"block5_conv4/Conv2D/ReadVariableOp
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0µ
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides

#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Î
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides

#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@¬
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@*
ksize
*
paddingVALID*
strides

"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ê
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides

#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdds
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides

#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdds
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd­
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
ksize
*
paddingVALID*
strides

"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22s
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22s
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22s
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22s
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22­
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block4_conv4/Conv2D/ReadVariableOpReadVariableOp+block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block4_conv4/Conv2DConv2Dblock4_conv3/Relu:activations:0*block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv4/BiasAdd/ReadVariableOpReadVariableOp,block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv4/BiasAddBiasAddblock4_conv4/Conv2D:output:0+block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv4/ReluRelublock4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
block4_pool/MaxPoolMaxPoolblock4_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block5_conv4/Conv2D/ReadVariableOpReadVariableOp+block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block5_conv4/Conv2DConv2Dblock5_conv3/Relu:activations:0*block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv4/BiasAdd/ReadVariableOpReadVariableOp,block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv4/BiasAddBiasAddblock5_conv4/Conv2D:output:0+block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv4/ReluRelublock5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
block5_pool/MaxPoolMaxPoolblock5_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
t
IdentityIdentityblock5_pool/MaxPool:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö	
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block3_conv4/BiasAdd/ReadVariableOp#^block3_conv4/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block4_conv4/BiasAdd/ReadVariableOp#^block4_conv4/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp$^block5_conv4/BiasAdd/ReadVariableOp#^block5_conv4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block3_conv4/BiasAdd/ReadVariableOp#block3_conv4/BiasAdd/ReadVariableOp2H
"block3_conv4/Conv2D/ReadVariableOp"block3_conv4/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block4_conv4/BiasAdd/ReadVariableOp#block4_conv4/BiasAdd/ReadVariableOp2H
"block4_conv4/Conv2D/ReadVariableOp"block4_conv4/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp2J
#block5_conv4/BiasAdd/ReadVariableOp#block5_conv4/BiasAdd/ReadVariableOp2H
"block5_conv4/Conv2D/ReadVariableOp"block5_conv4/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs
³`

__inference__traced_save_979406
file_prefix.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop2
.savev2_block1_conv1_kernel_read_readvariableop0
,savev2_block1_conv1_bias_read_readvariableop2
.savev2_block1_conv2_kernel_read_readvariableop0
,savev2_block1_conv2_bias_read_readvariableop2
.savev2_block2_conv1_kernel_read_readvariableop0
,savev2_block2_conv1_bias_read_readvariableop2
.savev2_block2_conv2_kernel_read_readvariableop0
,savev2_block2_conv2_bias_read_readvariableop2
.savev2_block3_conv1_kernel_read_readvariableop0
,savev2_block3_conv1_bias_read_readvariableop2
.savev2_block3_conv2_kernel_read_readvariableop0
,savev2_block3_conv2_bias_read_readvariableop2
.savev2_block3_conv3_kernel_read_readvariableop0
,savev2_block3_conv3_bias_read_readvariableop2
.savev2_block3_conv4_kernel_read_readvariableop0
,savev2_block3_conv4_bias_read_readvariableop2
.savev2_block4_conv1_kernel_read_readvariableop0
,savev2_block4_conv1_bias_read_readvariableop2
.savev2_block4_conv2_kernel_read_readvariableop0
,savev2_block4_conv2_bias_read_readvariableop2
.savev2_block4_conv3_kernel_read_readvariableop0
,savev2_block4_conv3_bias_read_readvariableop2
.savev2_block4_conv4_kernel_read_readvariableop0
,savev2_block4_conv4_bias_read_readvariableop2
.savev2_block5_conv1_kernel_read_readvariableop0
,savev2_block5_conv1_bias_read_readvariableop2
.savev2_block5_conv2_kernel_read_readvariableop0
,savev2_block5_conv2_bias_read_readvariableop2
.savev2_block5_conv3_kernel_read_readvariableop0
,savev2_block5_conv3_bias_read_readvariableop2
.savev2_block5_conv4_kernel_read_readvariableop0
,savev2_block5_conv4_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_14_kernel_m_read_readvariableop3
/savev2_adam_dense_14_bias_m_read_readvariableop5
1savev2_adam_dense_13_kernel_m_read_readvariableop3
/savev2_adam_dense_13_bias_m_read_readvariableop5
1savev2_adam_dense_14_kernel_v_read_readvariableop3
/savev2_adam_dense_14_bias_v_read_readvariableop5
1savev2_adam_dense_13_kernel_v_read_readvariableop3
/savev2_adam_dense_13_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: É
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*ò
valueèBå5B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH×
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ©
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop.savev2_block1_conv1_kernel_read_readvariableop,savev2_block1_conv1_bias_read_readvariableop.savev2_block1_conv2_kernel_read_readvariableop,savev2_block1_conv2_bias_read_readvariableop.savev2_block2_conv1_kernel_read_readvariableop,savev2_block2_conv1_bias_read_readvariableop.savev2_block2_conv2_kernel_read_readvariableop,savev2_block2_conv2_bias_read_readvariableop.savev2_block3_conv1_kernel_read_readvariableop,savev2_block3_conv1_bias_read_readvariableop.savev2_block3_conv2_kernel_read_readvariableop,savev2_block3_conv2_bias_read_readvariableop.savev2_block3_conv3_kernel_read_readvariableop,savev2_block3_conv3_bias_read_readvariableop.savev2_block3_conv4_kernel_read_readvariableop,savev2_block3_conv4_bias_read_readvariableop.savev2_block4_conv1_kernel_read_readvariableop,savev2_block4_conv1_bias_read_readvariableop.savev2_block4_conv2_kernel_read_readvariableop,savev2_block4_conv2_bias_read_readvariableop.savev2_block4_conv3_kernel_read_readvariableop,savev2_block4_conv3_bias_read_readvariableop.savev2_block4_conv4_kernel_read_readvariableop,savev2_block4_conv4_bias_read_readvariableop.savev2_block5_conv1_kernel_read_readvariableop,savev2_block5_conv1_bias_read_readvariableop.savev2_block5_conv2_kernel_read_readvariableop,savev2_block5_conv2_bias_read_readvariableop.savev2_block5_conv3_kernel_read_readvariableop,savev2_block5_conv3_bias_read_readvariableop.savev2_block5_conv4_kernel_read_readvariableop,savev2_block5_conv4_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_14_kernel_m_read_readvariableop/savev2_adam_dense_14_bias_m_read_readvariableop1savev2_adam_dense_13_kernel_m_read_readvariableop/savev2_adam_dense_13_bias_m_read_readvariableop1savev2_adam_dense_14_kernel_v_read_readvariableop/savev2_adam_dense_14_bias_v_read_readvariableop1savev2_adam_dense_13_kernel_v_read_readvariableop/savev2_adam_dense_13_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *C
dtypes9
725	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*µ
_input_shapes£
 : ::: : : : :@:@:@@:@:@::::::::::::::::::::::::::::	@:@: : : : :::	@:@:::	@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 

_output_shapes
:@:,	(
&
_output_shapes
:@@: 


_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::! 

_output_shapes	
::.!*
(
_output_shapes
::!"

_output_shapes	
::.#*
(
_output_shapes
::!$

_output_shapes	
::.%*
(
_output_shapes
::!&

_output_shapes	
::%'!

_output_shapes
:	@: (

_output_shapes
:@:)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :$- 

_output_shapes

:: .

_output_shapes
::%/!

_output_shapes
:	@: 0

_output_shapes
:@:$1 

_output_shapes

:: 2

_output_shapes
::%3!

_output_shapes
:	@: 4

_output_shapes
:@:5

_output_shapes
: 


H__inference_block4_conv3_layer_call_and_return_conditional_losses_975007

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block2_conv2_layer_call_and_return_conditional_losses_974886

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿdd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs


H__inference_block4_conv4_layer_call_and_return_conditional_losses_975024

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

c
G__inference_block5_pool_layer_call_and_return_conditional_losses_974813

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë	
ö
D__inference_dense_13_layer_call_and_return_conditional_losses_978856

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
¥
-__inference_block5_conv1_layer_call_fn_979145

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv1_layer_call_and_return_conditional_losses_975042x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block3_conv3_layer_call_and_return_conditional_losses_979016

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs


H__inference_block3_conv2_layer_call_and_return_conditional_losses_974921

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
ü
¥
-__inference_block3_conv3_layer_call_fn_979005

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv3_layer_call_and_return_conditional_losses_974938x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ22: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
ý
¢
-__inference_block1_conv2_layer_call_fn_978885

inputs!
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block1_conv2_layer_call_and_return_conditional_losses_974851y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÈÈ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
 
_user_specified_nameinputs
»ç
;
!__inference__wrapped_model_974756
input_27
input_28]
Cmodel_14_model_13_vgg19_block1_conv1_conv2d_readvariableop_resource:@R
Dmodel_14_model_13_vgg19_block1_conv1_biasadd_readvariableop_resource:@]
Cmodel_14_model_13_vgg19_block1_conv2_conv2d_readvariableop_resource:@@R
Dmodel_14_model_13_vgg19_block1_conv2_biasadd_readvariableop_resource:@^
Cmodel_14_model_13_vgg19_block2_conv1_conv2d_readvariableop_resource:@S
Dmodel_14_model_13_vgg19_block2_conv1_biasadd_readvariableop_resource:	_
Cmodel_14_model_13_vgg19_block2_conv2_conv2d_readvariableop_resource:S
Dmodel_14_model_13_vgg19_block2_conv2_biasadd_readvariableop_resource:	_
Cmodel_14_model_13_vgg19_block3_conv1_conv2d_readvariableop_resource:S
Dmodel_14_model_13_vgg19_block3_conv1_biasadd_readvariableop_resource:	_
Cmodel_14_model_13_vgg19_block3_conv2_conv2d_readvariableop_resource:S
Dmodel_14_model_13_vgg19_block3_conv2_biasadd_readvariableop_resource:	_
Cmodel_14_model_13_vgg19_block3_conv3_conv2d_readvariableop_resource:S
Dmodel_14_model_13_vgg19_block3_conv3_biasadd_readvariableop_resource:	_
Cmodel_14_model_13_vgg19_block3_conv4_conv2d_readvariableop_resource:S
Dmodel_14_model_13_vgg19_block3_conv4_biasadd_readvariableop_resource:	_
Cmodel_14_model_13_vgg19_block4_conv1_conv2d_readvariableop_resource:S
Dmodel_14_model_13_vgg19_block4_conv1_biasadd_readvariableop_resource:	_
Cmodel_14_model_13_vgg19_block4_conv2_conv2d_readvariableop_resource:S
Dmodel_14_model_13_vgg19_block4_conv2_biasadd_readvariableop_resource:	_
Cmodel_14_model_13_vgg19_block4_conv3_conv2d_readvariableop_resource:S
Dmodel_14_model_13_vgg19_block4_conv3_biasadd_readvariableop_resource:	_
Cmodel_14_model_13_vgg19_block4_conv4_conv2d_readvariableop_resource:S
Dmodel_14_model_13_vgg19_block4_conv4_biasadd_readvariableop_resource:	_
Cmodel_14_model_13_vgg19_block5_conv1_conv2d_readvariableop_resource:S
Dmodel_14_model_13_vgg19_block5_conv1_biasadd_readvariableop_resource:	_
Cmodel_14_model_13_vgg19_block5_conv2_conv2d_readvariableop_resource:S
Dmodel_14_model_13_vgg19_block5_conv2_biasadd_readvariableop_resource:	_
Cmodel_14_model_13_vgg19_block5_conv3_conv2d_readvariableop_resource:S
Dmodel_14_model_13_vgg19_block5_conv3_biasadd_readvariableop_resource:	_
Cmodel_14_model_13_vgg19_block5_conv4_conv2d_readvariableop_resource:S
Dmodel_14_model_13_vgg19_block5_conv4_biasadd_readvariableop_resource:	L
9model_14_model_13_dense_13_matmul_readvariableop_resource:	@H
:model_14_model_13_dense_13_biasadd_readvariableop_resource:@B
0model_14_dense_14_matmul_readvariableop_resource:?
1model_14_dense_14_biasadd_readvariableop_resource:
identity¢(model_14/dense_14/BiasAdd/ReadVariableOp¢'model_14/dense_14/MatMul/ReadVariableOp¢1model_14/model_13/dense_13/BiasAdd/ReadVariableOp¢3model_14/model_13/dense_13/BiasAdd_1/ReadVariableOp¢0model_14/model_13/dense_13/MatMul/ReadVariableOp¢2model_14/model_13/dense_13/MatMul_1/ReadVariableOp¢;model_14/model_13/vgg19/block1_conv1/BiasAdd/ReadVariableOp¢=model_14/model_13/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp¢:model_14/model_13/vgg19/block1_conv1/Conv2D/ReadVariableOp¢<model_14/model_13/vgg19/block1_conv1/Conv2D_1/ReadVariableOp¢;model_14/model_13/vgg19/block1_conv2/BiasAdd/ReadVariableOp¢=model_14/model_13/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp¢:model_14/model_13/vgg19/block1_conv2/Conv2D/ReadVariableOp¢<model_14/model_13/vgg19/block1_conv2/Conv2D_1/ReadVariableOp¢;model_14/model_13/vgg19/block2_conv1/BiasAdd/ReadVariableOp¢=model_14/model_13/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp¢:model_14/model_13/vgg19/block2_conv1/Conv2D/ReadVariableOp¢<model_14/model_13/vgg19/block2_conv1/Conv2D_1/ReadVariableOp¢;model_14/model_13/vgg19/block2_conv2/BiasAdd/ReadVariableOp¢=model_14/model_13/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp¢:model_14/model_13/vgg19/block2_conv2/Conv2D/ReadVariableOp¢<model_14/model_13/vgg19/block2_conv2/Conv2D_1/ReadVariableOp¢;model_14/model_13/vgg19/block3_conv1/BiasAdd/ReadVariableOp¢=model_14/model_13/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp¢:model_14/model_13/vgg19/block3_conv1/Conv2D/ReadVariableOp¢<model_14/model_13/vgg19/block3_conv1/Conv2D_1/ReadVariableOp¢;model_14/model_13/vgg19/block3_conv2/BiasAdd/ReadVariableOp¢=model_14/model_13/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp¢:model_14/model_13/vgg19/block3_conv2/Conv2D/ReadVariableOp¢<model_14/model_13/vgg19/block3_conv2/Conv2D_1/ReadVariableOp¢;model_14/model_13/vgg19/block3_conv3/BiasAdd/ReadVariableOp¢=model_14/model_13/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp¢:model_14/model_13/vgg19/block3_conv3/Conv2D/ReadVariableOp¢<model_14/model_13/vgg19/block3_conv3/Conv2D_1/ReadVariableOp¢;model_14/model_13/vgg19/block3_conv4/BiasAdd/ReadVariableOp¢=model_14/model_13/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp¢:model_14/model_13/vgg19/block3_conv4/Conv2D/ReadVariableOp¢<model_14/model_13/vgg19/block3_conv4/Conv2D_1/ReadVariableOp¢;model_14/model_13/vgg19/block4_conv1/BiasAdd/ReadVariableOp¢=model_14/model_13/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp¢:model_14/model_13/vgg19/block4_conv1/Conv2D/ReadVariableOp¢<model_14/model_13/vgg19/block4_conv1/Conv2D_1/ReadVariableOp¢;model_14/model_13/vgg19/block4_conv2/BiasAdd/ReadVariableOp¢=model_14/model_13/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp¢:model_14/model_13/vgg19/block4_conv2/Conv2D/ReadVariableOp¢<model_14/model_13/vgg19/block4_conv2/Conv2D_1/ReadVariableOp¢;model_14/model_13/vgg19/block4_conv3/BiasAdd/ReadVariableOp¢=model_14/model_13/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp¢:model_14/model_13/vgg19/block4_conv3/Conv2D/ReadVariableOp¢<model_14/model_13/vgg19/block4_conv3/Conv2D_1/ReadVariableOp¢;model_14/model_13/vgg19/block4_conv4/BiasAdd/ReadVariableOp¢=model_14/model_13/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp¢:model_14/model_13/vgg19/block4_conv4/Conv2D/ReadVariableOp¢<model_14/model_13/vgg19/block4_conv4/Conv2D_1/ReadVariableOp¢;model_14/model_13/vgg19/block5_conv1/BiasAdd/ReadVariableOp¢=model_14/model_13/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp¢:model_14/model_13/vgg19/block5_conv1/Conv2D/ReadVariableOp¢<model_14/model_13/vgg19/block5_conv1/Conv2D_1/ReadVariableOp¢;model_14/model_13/vgg19/block5_conv2/BiasAdd/ReadVariableOp¢=model_14/model_13/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp¢:model_14/model_13/vgg19/block5_conv2/Conv2D/ReadVariableOp¢<model_14/model_13/vgg19/block5_conv2/Conv2D_1/ReadVariableOp¢;model_14/model_13/vgg19/block5_conv3/BiasAdd/ReadVariableOp¢=model_14/model_13/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp¢:model_14/model_13/vgg19/block5_conv3/Conv2D/ReadVariableOp¢<model_14/model_13/vgg19/block5_conv3/Conv2D_1/ReadVariableOp¢;model_14/model_13/vgg19/block5_conv4/BiasAdd/ReadVariableOp¢=model_14/model_13/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp¢:model_14/model_13/vgg19/block5_conv4/Conv2D/ReadVariableOp¢<model_14/model_13/vgg19/block5_conv4/Conv2D_1/ReadVariableOpÆ
:model_14/model_13/vgg19/block1_conv1/Conv2D/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0ç
+model_14/model_13/vgg19/block1_conv1/Conv2DConv2Dinput_27Bmodel_14/model_13/vgg19/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
¼
;model_14/model_13/vgg19/block1_conv1/BiasAdd/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0î
,model_14/model_13/vgg19/block1_conv1/BiasAddBiasAdd4model_14/model_13/vgg19/block1_conv1/Conv2D:output:0Cmodel_14/model_13/vgg19/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@¤
)model_14/model_13/vgg19/block1_conv1/ReluRelu5model_14/model_13/vgg19/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@Æ
:model_14/model_13/vgg19/block1_conv2/Conv2D/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
+model_14/model_13/vgg19/block1_conv2/Conv2DConv2D7model_14/model_13/vgg19/block1_conv1/Relu:activations:0Bmodel_14/model_13/vgg19/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
¼
;model_14/model_13/vgg19/block1_conv2/BiasAdd/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0î
,model_14/model_13/vgg19/block1_conv2/BiasAddBiasAdd4model_14/model_13/vgg19/block1_conv2/Conv2D:output:0Cmodel_14/model_13/vgg19/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@¤
)model_14/model_13/vgg19/block1_conv2/ReluRelu5model_14/model_13/vgg19/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@Ü
+model_14/model_13/vgg19/block1_pool/MaxPoolMaxPool7model_14/model_13/vgg19/block1_conv2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@*
ksize
*
paddingVALID*
strides
Ç
:model_14/model_13/vgg19/block2_conv1/Conv2D/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
+model_14/model_13/vgg19/block2_conv1/Conv2DConv2D4model_14/model_13/vgg19/block1_pool/MaxPool:output:0Bmodel_14/model_13/vgg19/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
½
;model_14/model_13/vgg19/block2_conv1/BiasAdd/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_14/model_13/vgg19/block2_conv1/BiasAddBiasAdd4model_14/model_13/vgg19/block2_conv1/Conv2D:output:0Cmodel_14/model_13/vgg19/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd£
)model_14/model_13/vgg19/block2_conv1/ReluRelu5model_14/model_13/vgg19/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddÈ
:model_14/model_13/vgg19/block2_conv2/Conv2D/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
+model_14/model_13/vgg19/block2_conv2/Conv2DConv2D7model_14/model_13/vgg19/block2_conv1/Relu:activations:0Bmodel_14/model_13/vgg19/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
½
;model_14/model_13/vgg19/block2_conv2/BiasAdd/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_14/model_13/vgg19/block2_conv2/BiasAddBiasAdd4model_14/model_13/vgg19/block2_conv2/Conv2D:output:0Cmodel_14/model_13/vgg19/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd£
)model_14/model_13/vgg19/block2_conv2/ReluRelu5model_14/model_13/vgg19/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddÝ
+model_14/model_13/vgg19/block2_pool/MaxPoolMaxPool7model_14/model_13/vgg19/block2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
ksize
*
paddingVALID*
strides
È
:model_14/model_13/vgg19/block3_conv1/Conv2D/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
+model_14/model_13/vgg19/block3_conv1/Conv2DConv2D4model_14/model_13/vgg19/block2_pool/MaxPool:output:0Bmodel_14/model_13/vgg19/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
½
;model_14/model_13/vgg19/block3_conv1/BiasAdd/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_14/model_13/vgg19/block3_conv1/BiasAddBiasAdd4model_14/model_13/vgg19/block3_conv1/Conv2D:output:0Cmodel_14/model_13/vgg19/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22£
)model_14/model_13/vgg19/block3_conv1/ReluRelu5model_14/model_13/vgg19/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22È
:model_14/model_13/vgg19/block3_conv2/Conv2D/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
+model_14/model_13/vgg19/block3_conv2/Conv2DConv2D7model_14/model_13/vgg19/block3_conv1/Relu:activations:0Bmodel_14/model_13/vgg19/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
½
;model_14/model_13/vgg19/block3_conv2/BiasAdd/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_14/model_13/vgg19/block3_conv2/BiasAddBiasAdd4model_14/model_13/vgg19/block3_conv2/Conv2D:output:0Cmodel_14/model_13/vgg19/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22£
)model_14/model_13/vgg19/block3_conv2/ReluRelu5model_14/model_13/vgg19/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22È
:model_14/model_13/vgg19/block3_conv3/Conv2D/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
+model_14/model_13/vgg19/block3_conv3/Conv2DConv2D7model_14/model_13/vgg19/block3_conv2/Relu:activations:0Bmodel_14/model_13/vgg19/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
½
;model_14/model_13/vgg19/block3_conv3/BiasAdd/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_14/model_13/vgg19/block3_conv3/BiasAddBiasAdd4model_14/model_13/vgg19/block3_conv3/Conv2D:output:0Cmodel_14/model_13/vgg19/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22£
)model_14/model_13/vgg19/block3_conv3/ReluRelu5model_14/model_13/vgg19/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22È
:model_14/model_13/vgg19/block3_conv4/Conv2D/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
+model_14/model_13/vgg19/block3_conv4/Conv2DConv2D7model_14/model_13/vgg19/block3_conv3/Relu:activations:0Bmodel_14/model_13/vgg19/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
½
;model_14/model_13/vgg19/block3_conv4/BiasAdd/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_14/model_13/vgg19/block3_conv4/BiasAddBiasAdd4model_14/model_13/vgg19/block3_conv4/Conv2D:output:0Cmodel_14/model_13/vgg19/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22£
)model_14/model_13/vgg19/block3_conv4/ReluRelu5model_14/model_13/vgg19/block3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Ý
+model_14/model_13/vgg19/block3_pool/MaxPoolMaxPool7model_14/model_13/vgg19/block3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
È
:model_14/model_13/vgg19/block4_conv1/Conv2D/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
+model_14/model_13/vgg19/block4_conv1/Conv2DConv2D4model_14/model_13/vgg19/block3_pool/MaxPool:output:0Bmodel_14/model_13/vgg19/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
½
;model_14/model_13/vgg19/block4_conv1/BiasAdd/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_14/model_13/vgg19/block4_conv1/BiasAddBiasAdd4model_14/model_13/vgg19/block4_conv1/Conv2D:output:0Cmodel_14/model_13/vgg19/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
)model_14/model_13/vgg19/block4_conv1/ReluRelu5model_14/model_13/vgg19/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
:model_14/model_13/vgg19/block4_conv2/Conv2D/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
+model_14/model_13/vgg19/block4_conv2/Conv2DConv2D7model_14/model_13/vgg19/block4_conv1/Relu:activations:0Bmodel_14/model_13/vgg19/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
½
;model_14/model_13/vgg19/block4_conv2/BiasAdd/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_14/model_13/vgg19/block4_conv2/BiasAddBiasAdd4model_14/model_13/vgg19/block4_conv2/Conv2D:output:0Cmodel_14/model_13/vgg19/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
)model_14/model_13/vgg19/block4_conv2/ReluRelu5model_14/model_13/vgg19/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
:model_14/model_13/vgg19/block4_conv3/Conv2D/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
+model_14/model_13/vgg19/block4_conv3/Conv2DConv2D7model_14/model_13/vgg19/block4_conv2/Relu:activations:0Bmodel_14/model_13/vgg19/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
½
;model_14/model_13/vgg19/block4_conv3/BiasAdd/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_14/model_13/vgg19/block4_conv3/BiasAddBiasAdd4model_14/model_13/vgg19/block4_conv3/Conv2D:output:0Cmodel_14/model_13/vgg19/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
)model_14/model_13/vgg19/block4_conv3/ReluRelu5model_14/model_13/vgg19/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
:model_14/model_13/vgg19/block4_conv4/Conv2D/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
+model_14/model_13/vgg19/block4_conv4/Conv2DConv2D7model_14/model_13/vgg19/block4_conv3/Relu:activations:0Bmodel_14/model_13/vgg19/block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
½
;model_14/model_13/vgg19/block4_conv4/BiasAdd/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_14/model_13/vgg19/block4_conv4/BiasAddBiasAdd4model_14/model_13/vgg19/block4_conv4/Conv2D:output:0Cmodel_14/model_13/vgg19/block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
)model_14/model_13/vgg19/block4_conv4/ReluRelu5model_14/model_13/vgg19/block4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
+model_14/model_13/vgg19/block4_pool/MaxPoolMaxPool7model_14/model_13/vgg19/block4_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
È
:model_14/model_13/vgg19/block5_conv1/Conv2D/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
+model_14/model_13/vgg19/block5_conv1/Conv2DConv2D4model_14/model_13/vgg19/block4_pool/MaxPool:output:0Bmodel_14/model_13/vgg19/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
½
;model_14/model_13/vgg19/block5_conv1/BiasAdd/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_14/model_13/vgg19/block5_conv1/BiasAddBiasAdd4model_14/model_13/vgg19/block5_conv1/Conv2D:output:0Cmodel_14/model_13/vgg19/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
)model_14/model_13/vgg19/block5_conv1/ReluRelu5model_14/model_13/vgg19/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
:model_14/model_13/vgg19/block5_conv2/Conv2D/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
+model_14/model_13/vgg19/block5_conv2/Conv2DConv2D7model_14/model_13/vgg19/block5_conv1/Relu:activations:0Bmodel_14/model_13/vgg19/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
½
;model_14/model_13/vgg19/block5_conv2/BiasAdd/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_14/model_13/vgg19/block5_conv2/BiasAddBiasAdd4model_14/model_13/vgg19/block5_conv2/Conv2D:output:0Cmodel_14/model_13/vgg19/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
)model_14/model_13/vgg19/block5_conv2/ReluRelu5model_14/model_13/vgg19/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
:model_14/model_13/vgg19/block5_conv3/Conv2D/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
+model_14/model_13/vgg19/block5_conv3/Conv2DConv2D7model_14/model_13/vgg19/block5_conv2/Relu:activations:0Bmodel_14/model_13/vgg19/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
½
;model_14/model_13/vgg19/block5_conv3/BiasAdd/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_14/model_13/vgg19/block5_conv3/BiasAddBiasAdd4model_14/model_13/vgg19/block5_conv3/Conv2D:output:0Cmodel_14/model_13/vgg19/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
)model_14/model_13/vgg19/block5_conv3/ReluRelu5model_14/model_13/vgg19/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
:model_14/model_13/vgg19/block5_conv4/Conv2D/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
+model_14/model_13/vgg19/block5_conv4/Conv2DConv2D7model_14/model_13/vgg19/block5_conv3/Relu:activations:0Bmodel_14/model_13/vgg19/block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
½
;model_14/model_13/vgg19/block5_conv4/BiasAdd/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_14/model_13/vgg19/block5_conv4/BiasAddBiasAdd4model_14/model_13/vgg19/block5_conv4/Conv2D:output:0Cmodel_14/model_13/vgg19/block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
)model_14/model_13/vgg19/block5_conv4/ReluRelu5model_14/model_13/vgg19/block5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
+model_14/model_13/vgg19/block5_pool/MaxPoolMaxPool7model_14/model_13/vgg19/block5_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

Cmodel_14/model_13/global_average_pooling2d_9/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ð
1model_14/model_13/global_average_pooling2d_9/MeanMean4model_14/model_13/vgg19/block5_pool/MaxPool:output:0Lmodel_14/model_13/global_average_pooling2d_9/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
0model_14/model_13/dense_13/MatMul/ReadVariableOpReadVariableOp9model_14_model_13_dense_13_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0Ó
!model_14/model_13/dense_13/MatMulMatMul:model_14/model_13/global_average_pooling2d_9/Mean:output:08model_14/model_13/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¨
1model_14/model_13/dense_13/BiasAdd/ReadVariableOpReadVariableOp:model_14_model_13_dense_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ç
"model_14/model_13/dense_13/BiasAddBiasAdd+model_14/model_13/dense_13/MatMul:product:09model_14/model_13/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@È
<model_14/model_13/vgg19/block1_conv1/Conv2D_1/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0ë
-model_14/model_13/vgg19/block1_conv1/Conv2D_1Conv2Dinput_28Dmodel_14/model_13/vgg19/block1_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
¾
=model_14/model_13/vgg19/block1_conv1/BiasAdd_1/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ô
.model_14/model_13/vgg19/block1_conv1/BiasAdd_1BiasAdd6model_14/model_13/vgg19/block1_conv1/Conv2D_1:output:0Emodel_14/model_13/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@¨
+model_14/model_13/vgg19/block1_conv1/Relu_1Relu7model_14/model_13/vgg19/block1_conv1/BiasAdd_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@È
<model_14/model_13/vgg19/block1_conv2/Conv2D_1/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
-model_14/model_13/vgg19/block1_conv2/Conv2D_1Conv2D9model_14/model_13/vgg19/block1_conv1/Relu_1:activations:0Dmodel_14/model_13/vgg19/block1_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
¾
=model_14/model_13/vgg19/block1_conv2/BiasAdd_1/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ô
.model_14/model_13/vgg19/block1_conv2/BiasAdd_1BiasAdd6model_14/model_13/vgg19/block1_conv2/Conv2D_1:output:0Emodel_14/model_13/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@¨
+model_14/model_13/vgg19/block1_conv2/Relu_1Relu7model_14/model_13/vgg19/block1_conv2/BiasAdd_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@à
-model_14/model_13/vgg19/block1_pool/MaxPool_1MaxPool9model_14/model_13/vgg19/block1_conv2/Relu_1:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@*
ksize
*
paddingVALID*
strides
É
<model_14/model_13/vgg19/block2_conv1/Conv2D_1/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
-model_14/model_13/vgg19/block2_conv1/Conv2D_1Conv2D6model_14/model_13/vgg19/block1_pool/MaxPool_1:output:0Dmodel_14/model_13/vgg19/block2_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
¿
=model_14/model_13/vgg19/block2_conv1/BiasAdd_1/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ó
.model_14/model_13/vgg19/block2_conv1/BiasAdd_1BiasAdd6model_14/model_13/vgg19/block2_conv1/Conv2D_1:output:0Emodel_14/model_13/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd§
+model_14/model_13/vgg19/block2_conv1/Relu_1Relu7model_14/model_13/vgg19/block2_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddÊ
<model_14/model_13/vgg19/block2_conv2/Conv2D_1/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
-model_14/model_13/vgg19/block2_conv2/Conv2D_1Conv2D9model_14/model_13/vgg19/block2_conv1/Relu_1:activations:0Dmodel_14/model_13/vgg19/block2_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
¿
=model_14/model_13/vgg19/block2_conv2/BiasAdd_1/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ó
.model_14/model_13/vgg19/block2_conv2/BiasAdd_1BiasAdd6model_14/model_13/vgg19/block2_conv2/Conv2D_1:output:0Emodel_14/model_13/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd§
+model_14/model_13/vgg19/block2_conv2/Relu_1Relu7model_14/model_13/vgg19/block2_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddá
-model_14/model_13/vgg19/block2_pool/MaxPool_1MaxPool9model_14/model_13/vgg19/block2_conv2/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
ksize
*
paddingVALID*
strides
Ê
<model_14/model_13/vgg19/block3_conv1/Conv2D_1/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
-model_14/model_13/vgg19/block3_conv1/Conv2D_1Conv2D6model_14/model_13/vgg19/block2_pool/MaxPool_1:output:0Dmodel_14/model_13/vgg19/block3_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
¿
=model_14/model_13/vgg19/block3_conv1/BiasAdd_1/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ó
.model_14/model_13/vgg19/block3_conv1/BiasAdd_1BiasAdd6model_14/model_13/vgg19/block3_conv1/Conv2D_1:output:0Emodel_14/model_13/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22§
+model_14/model_13/vgg19/block3_conv1/Relu_1Relu7model_14/model_13/vgg19/block3_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Ê
<model_14/model_13/vgg19/block3_conv2/Conv2D_1/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
-model_14/model_13/vgg19/block3_conv2/Conv2D_1Conv2D9model_14/model_13/vgg19/block3_conv1/Relu_1:activations:0Dmodel_14/model_13/vgg19/block3_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
¿
=model_14/model_13/vgg19/block3_conv2/BiasAdd_1/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ó
.model_14/model_13/vgg19/block3_conv2/BiasAdd_1BiasAdd6model_14/model_13/vgg19/block3_conv2/Conv2D_1:output:0Emodel_14/model_13/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22§
+model_14/model_13/vgg19/block3_conv2/Relu_1Relu7model_14/model_13/vgg19/block3_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Ê
<model_14/model_13/vgg19/block3_conv3/Conv2D_1/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
-model_14/model_13/vgg19/block3_conv3/Conv2D_1Conv2D9model_14/model_13/vgg19/block3_conv2/Relu_1:activations:0Dmodel_14/model_13/vgg19/block3_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
¿
=model_14/model_13/vgg19/block3_conv3/BiasAdd_1/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ó
.model_14/model_13/vgg19/block3_conv3/BiasAdd_1BiasAdd6model_14/model_13/vgg19/block3_conv3/Conv2D_1:output:0Emodel_14/model_13/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22§
+model_14/model_13/vgg19/block3_conv3/Relu_1Relu7model_14/model_13/vgg19/block3_conv3/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Ê
<model_14/model_13/vgg19/block3_conv4/Conv2D_1/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
-model_14/model_13/vgg19/block3_conv4/Conv2D_1Conv2D9model_14/model_13/vgg19/block3_conv3/Relu_1:activations:0Dmodel_14/model_13/vgg19/block3_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
¿
=model_14/model_13/vgg19/block3_conv4/BiasAdd_1/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ó
.model_14/model_13/vgg19/block3_conv4/BiasAdd_1BiasAdd6model_14/model_13/vgg19/block3_conv4/Conv2D_1:output:0Emodel_14/model_13/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22§
+model_14/model_13/vgg19/block3_conv4/Relu_1Relu7model_14/model_13/vgg19/block3_conv4/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22á
-model_14/model_13/vgg19/block3_pool/MaxPool_1MaxPool9model_14/model_13/vgg19/block3_conv4/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
Ê
<model_14/model_13/vgg19/block4_conv1/Conv2D_1/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
-model_14/model_13/vgg19/block4_conv1/Conv2D_1Conv2D6model_14/model_13/vgg19/block3_pool/MaxPool_1:output:0Dmodel_14/model_13/vgg19/block4_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¿
=model_14/model_13/vgg19/block4_conv1/BiasAdd_1/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ó
.model_14/model_13/vgg19/block4_conv1/BiasAdd_1BiasAdd6model_14/model_13/vgg19/block4_conv1/Conv2D_1:output:0Emodel_14/model_13/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
+model_14/model_13/vgg19/block4_conv1/Relu_1Relu7model_14/model_13/vgg19/block4_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
<model_14/model_13/vgg19/block4_conv2/Conv2D_1/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
-model_14/model_13/vgg19/block4_conv2/Conv2D_1Conv2D9model_14/model_13/vgg19/block4_conv1/Relu_1:activations:0Dmodel_14/model_13/vgg19/block4_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¿
=model_14/model_13/vgg19/block4_conv2/BiasAdd_1/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ó
.model_14/model_13/vgg19/block4_conv2/BiasAdd_1BiasAdd6model_14/model_13/vgg19/block4_conv2/Conv2D_1:output:0Emodel_14/model_13/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
+model_14/model_13/vgg19/block4_conv2/Relu_1Relu7model_14/model_13/vgg19/block4_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
<model_14/model_13/vgg19/block4_conv3/Conv2D_1/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
-model_14/model_13/vgg19/block4_conv3/Conv2D_1Conv2D9model_14/model_13/vgg19/block4_conv2/Relu_1:activations:0Dmodel_14/model_13/vgg19/block4_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¿
=model_14/model_13/vgg19/block4_conv3/BiasAdd_1/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ó
.model_14/model_13/vgg19/block4_conv3/BiasAdd_1BiasAdd6model_14/model_13/vgg19/block4_conv3/Conv2D_1:output:0Emodel_14/model_13/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
+model_14/model_13/vgg19/block4_conv3/Relu_1Relu7model_14/model_13/vgg19/block4_conv3/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
<model_14/model_13/vgg19/block4_conv4/Conv2D_1/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
-model_14/model_13/vgg19/block4_conv4/Conv2D_1Conv2D9model_14/model_13/vgg19/block4_conv3/Relu_1:activations:0Dmodel_14/model_13/vgg19/block4_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¿
=model_14/model_13/vgg19/block4_conv4/BiasAdd_1/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ó
.model_14/model_13/vgg19/block4_conv4/BiasAdd_1BiasAdd6model_14/model_13/vgg19/block4_conv4/Conv2D_1:output:0Emodel_14/model_13/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
+model_14/model_13/vgg19/block4_conv4/Relu_1Relu7model_14/model_13/vgg19/block4_conv4/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿá
-model_14/model_13/vgg19/block4_pool/MaxPool_1MaxPool9model_14/model_13/vgg19/block4_conv4/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
Ê
<model_14/model_13/vgg19/block5_conv1/Conv2D_1/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
-model_14/model_13/vgg19/block5_conv1/Conv2D_1Conv2D6model_14/model_13/vgg19/block4_pool/MaxPool_1:output:0Dmodel_14/model_13/vgg19/block5_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¿
=model_14/model_13/vgg19/block5_conv1/BiasAdd_1/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ó
.model_14/model_13/vgg19/block5_conv1/BiasAdd_1BiasAdd6model_14/model_13/vgg19/block5_conv1/Conv2D_1:output:0Emodel_14/model_13/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
+model_14/model_13/vgg19/block5_conv1/Relu_1Relu7model_14/model_13/vgg19/block5_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
<model_14/model_13/vgg19/block5_conv2/Conv2D_1/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
-model_14/model_13/vgg19/block5_conv2/Conv2D_1Conv2D9model_14/model_13/vgg19/block5_conv1/Relu_1:activations:0Dmodel_14/model_13/vgg19/block5_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¿
=model_14/model_13/vgg19/block5_conv2/BiasAdd_1/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ó
.model_14/model_13/vgg19/block5_conv2/BiasAdd_1BiasAdd6model_14/model_13/vgg19/block5_conv2/Conv2D_1:output:0Emodel_14/model_13/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
+model_14/model_13/vgg19/block5_conv2/Relu_1Relu7model_14/model_13/vgg19/block5_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
<model_14/model_13/vgg19/block5_conv3/Conv2D_1/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
-model_14/model_13/vgg19/block5_conv3/Conv2D_1Conv2D9model_14/model_13/vgg19/block5_conv2/Relu_1:activations:0Dmodel_14/model_13/vgg19/block5_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¿
=model_14/model_13/vgg19/block5_conv3/BiasAdd_1/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ó
.model_14/model_13/vgg19/block5_conv3/BiasAdd_1BiasAdd6model_14/model_13/vgg19/block5_conv3/Conv2D_1:output:0Emodel_14/model_13/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
+model_14/model_13/vgg19/block5_conv3/Relu_1Relu7model_14/model_13/vgg19/block5_conv3/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
<model_14/model_13/vgg19/block5_conv4/Conv2D_1/ReadVariableOpReadVariableOpCmodel_14_model_13_vgg19_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
-model_14/model_13/vgg19/block5_conv4/Conv2D_1Conv2D9model_14/model_13/vgg19/block5_conv3/Relu_1:activations:0Dmodel_14/model_13/vgg19/block5_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¿
=model_14/model_13/vgg19/block5_conv4/BiasAdd_1/ReadVariableOpReadVariableOpDmodel_14_model_13_vgg19_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ó
.model_14/model_13/vgg19/block5_conv4/BiasAdd_1BiasAdd6model_14/model_13/vgg19/block5_conv4/Conv2D_1:output:0Emodel_14/model_13/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
+model_14/model_13/vgg19/block5_conv4/Relu_1Relu7model_14/model_13/vgg19/block5_conv4/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿá
-model_14/model_13/vgg19/block5_pool/MaxPool_1MaxPool9model_14/model_13/vgg19/block5_conv4/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

Emodel_14/model_13/global_average_pooling2d_9/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ö
3model_14/model_13/global_average_pooling2d_9/Mean_1Mean6model_14/model_13/vgg19/block5_pool/MaxPool_1:output:0Nmodel_14/model_13/global_average_pooling2d_9/Mean_1/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
2model_14/model_13/dense_13/MatMul_1/ReadVariableOpReadVariableOp9model_14_model_13_dense_13_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0Ù
#model_14/model_13/dense_13/MatMul_1MatMul<model_14/model_13/global_average_pooling2d_9/Mean_1:output:0:model_14/model_13/dense_13/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ª
3model_14/model_13/dense_13/BiasAdd_1/ReadVariableOpReadVariableOp:model_14_model_13_dense_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Í
$model_14/model_13/dense_13/BiasAdd_1BiasAdd-model_14/model_13/dense_13/MatMul_1:product:0;model_14/model_13/dense_13/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ª
model_14/lambda_4/subSub+model_14/model_13/dense_13/BiasAdd:output:0-model_14/model_13/dense_13/BiasAdd_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
model_14/lambda_4/SquareSquaremodel_14/lambda_4/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
'model_14/lambda_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :¯
model_14/lambda_4/SumSummodel_14/lambda_4/Square:y:00model_14/lambda_4/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(`
model_14/lambda_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
model_14/lambda_4/MaximumMaximummodel_14/lambda_4/Sum:output:0$model_14/lambda_4/Maximum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
model_14/lambda_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
model_14/lambda_4/Maximum_1Maximummodel_14/lambda_4/Maximum:z:0 model_14/lambda_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
model_14/lambda_4/SqrtSqrtmodel_14/lambda_4/Maximum_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model_14/dense_14/MatMul/ReadVariableOpReadVariableOp0model_14_dense_14_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¡
model_14/dense_14/MatMulMatMulmodel_14/lambda_4/Sqrt:y:0/model_14/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(model_14/dense_14/BiasAdd/ReadVariableOpReadVariableOp1model_14_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¬
model_14/dense_14/BiasAddBiasAdd"model_14/dense_14/MatMul:product:00model_14/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
model_14/dense_14/SigmoidSigmoid"model_14/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
IdentityIdentitymodel_14/dense_14/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
NoOpNoOp)^model_14/dense_14/BiasAdd/ReadVariableOp(^model_14/dense_14/MatMul/ReadVariableOp2^model_14/model_13/dense_13/BiasAdd/ReadVariableOp4^model_14/model_13/dense_13/BiasAdd_1/ReadVariableOp1^model_14/model_13/dense_13/MatMul/ReadVariableOp3^model_14/model_13/dense_13/MatMul_1/ReadVariableOp<^model_14/model_13/vgg19/block1_conv1/BiasAdd/ReadVariableOp>^model_14/model_13/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp;^model_14/model_13/vgg19/block1_conv1/Conv2D/ReadVariableOp=^model_14/model_13/vgg19/block1_conv1/Conv2D_1/ReadVariableOp<^model_14/model_13/vgg19/block1_conv2/BiasAdd/ReadVariableOp>^model_14/model_13/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp;^model_14/model_13/vgg19/block1_conv2/Conv2D/ReadVariableOp=^model_14/model_13/vgg19/block1_conv2/Conv2D_1/ReadVariableOp<^model_14/model_13/vgg19/block2_conv1/BiasAdd/ReadVariableOp>^model_14/model_13/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp;^model_14/model_13/vgg19/block2_conv1/Conv2D/ReadVariableOp=^model_14/model_13/vgg19/block2_conv1/Conv2D_1/ReadVariableOp<^model_14/model_13/vgg19/block2_conv2/BiasAdd/ReadVariableOp>^model_14/model_13/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp;^model_14/model_13/vgg19/block2_conv2/Conv2D/ReadVariableOp=^model_14/model_13/vgg19/block2_conv2/Conv2D_1/ReadVariableOp<^model_14/model_13/vgg19/block3_conv1/BiasAdd/ReadVariableOp>^model_14/model_13/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp;^model_14/model_13/vgg19/block3_conv1/Conv2D/ReadVariableOp=^model_14/model_13/vgg19/block3_conv1/Conv2D_1/ReadVariableOp<^model_14/model_13/vgg19/block3_conv2/BiasAdd/ReadVariableOp>^model_14/model_13/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp;^model_14/model_13/vgg19/block3_conv2/Conv2D/ReadVariableOp=^model_14/model_13/vgg19/block3_conv2/Conv2D_1/ReadVariableOp<^model_14/model_13/vgg19/block3_conv3/BiasAdd/ReadVariableOp>^model_14/model_13/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp;^model_14/model_13/vgg19/block3_conv3/Conv2D/ReadVariableOp=^model_14/model_13/vgg19/block3_conv3/Conv2D_1/ReadVariableOp<^model_14/model_13/vgg19/block3_conv4/BiasAdd/ReadVariableOp>^model_14/model_13/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp;^model_14/model_13/vgg19/block3_conv4/Conv2D/ReadVariableOp=^model_14/model_13/vgg19/block3_conv4/Conv2D_1/ReadVariableOp<^model_14/model_13/vgg19/block4_conv1/BiasAdd/ReadVariableOp>^model_14/model_13/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp;^model_14/model_13/vgg19/block4_conv1/Conv2D/ReadVariableOp=^model_14/model_13/vgg19/block4_conv1/Conv2D_1/ReadVariableOp<^model_14/model_13/vgg19/block4_conv2/BiasAdd/ReadVariableOp>^model_14/model_13/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp;^model_14/model_13/vgg19/block4_conv2/Conv2D/ReadVariableOp=^model_14/model_13/vgg19/block4_conv2/Conv2D_1/ReadVariableOp<^model_14/model_13/vgg19/block4_conv3/BiasAdd/ReadVariableOp>^model_14/model_13/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp;^model_14/model_13/vgg19/block4_conv3/Conv2D/ReadVariableOp=^model_14/model_13/vgg19/block4_conv3/Conv2D_1/ReadVariableOp<^model_14/model_13/vgg19/block4_conv4/BiasAdd/ReadVariableOp>^model_14/model_13/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp;^model_14/model_13/vgg19/block4_conv4/Conv2D/ReadVariableOp=^model_14/model_13/vgg19/block4_conv4/Conv2D_1/ReadVariableOp<^model_14/model_13/vgg19/block5_conv1/BiasAdd/ReadVariableOp>^model_14/model_13/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp;^model_14/model_13/vgg19/block5_conv1/Conv2D/ReadVariableOp=^model_14/model_13/vgg19/block5_conv1/Conv2D_1/ReadVariableOp<^model_14/model_13/vgg19/block5_conv2/BiasAdd/ReadVariableOp>^model_14/model_13/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp;^model_14/model_13/vgg19/block5_conv2/Conv2D/ReadVariableOp=^model_14/model_13/vgg19/block5_conv2/Conv2D_1/ReadVariableOp<^model_14/model_13/vgg19/block5_conv3/BiasAdd/ReadVariableOp>^model_14/model_13/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp;^model_14/model_13/vgg19/block5_conv3/Conv2D/ReadVariableOp=^model_14/model_13/vgg19/block5_conv3/Conv2D_1/ReadVariableOp<^model_14/model_13/vgg19/block5_conv4/BiasAdd/ReadVariableOp>^model_14/model_13/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp;^model_14/model_13/vgg19/block5_conv4/Conv2D/ReadVariableOp=^model_14/model_13/vgg19/block5_conv4/Conv2D_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2T
(model_14/dense_14/BiasAdd/ReadVariableOp(model_14/dense_14/BiasAdd/ReadVariableOp2R
'model_14/dense_14/MatMul/ReadVariableOp'model_14/dense_14/MatMul/ReadVariableOp2f
1model_14/model_13/dense_13/BiasAdd/ReadVariableOp1model_14/model_13/dense_13/BiasAdd/ReadVariableOp2j
3model_14/model_13/dense_13/BiasAdd_1/ReadVariableOp3model_14/model_13/dense_13/BiasAdd_1/ReadVariableOp2d
0model_14/model_13/dense_13/MatMul/ReadVariableOp0model_14/model_13/dense_13/MatMul/ReadVariableOp2h
2model_14/model_13/dense_13/MatMul_1/ReadVariableOp2model_14/model_13/dense_13/MatMul_1/ReadVariableOp2z
;model_14/model_13/vgg19/block1_conv1/BiasAdd/ReadVariableOp;model_14/model_13/vgg19/block1_conv1/BiasAdd/ReadVariableOp2~
=model_14/model_13/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp=model_14/model_13/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp2x
:model_14/model_13/vgg19/block1_conv1/Conv2D/ReadVariableOp:model_14/model_13/vgg19/block1_conv1/Conv2D/ReadVariableOp2|
<model_14/model_13/vgg19/block1_conv1/Conv2D_1/ReadVariableOp<model_14/model_13/vgg19/block1_conv1/Conv2D_1/ReadVariableOp2z
;model_14/model_13/vgg19/block1_conv2/BiasAdd/ReadVariableOp;model_14/model_13/vgg19/block1_conv2/BiasAdd/ReadVariableOp2~
=model_14/model_13/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp=model_14/model_13/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp2x
:model_14/model_13/vgg19/block1_conv2/Conv2D/ReadVariableOp:model_14/model_13/vgg19/block1_conv2/Conv2D/ReadVariableOp2|
<model_14/model_13/vgg19/block1_conv2/Conv2D_1/ReadVariableOp<model_14/model_13/vgg19/block1_conv2/Conv2D_1/ReadVariableOp2z
;model_14/model_13/vgg19/block2_conv1/BiasAdd/ReadVariableOp;model_14/model_13/vgg19/block2_conv1/BiasAdd/ReadVariableOp2~
=model_14/model_13/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp=model_14/model_13/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp2x
:model_14/model_13/vgg19/block2_conv1/Conv2D/ReadVariableOp:model_14/model_13/vgg19/block2_conv1/Conv2D/ReadVariableOp2|
<model_14/model_13/vgg19/block2_conv1/Conv2D_1/ReadVariableOp<model_14/model_13/vgg19/block2_conv1/Conv2D_1/ReadVariableOp2z
;model_14/model_13/vgg19/block2_conv2/BiasAdd/ReadVariableOp;model_14/model_13/vgg19/block2_conv2/BiasAdd/ReadVariableOp2~
=model_14/model_13/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp=model_14/model_13/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp2x
:model_14/model_13/vgg19/block2_conv2/Conv2D/ReadVariableOp:model_14/model_13/vgg19/block2_conv2/Conv2D/ReadVariableOp2|
<model_14/model_13/vgg19/block2_conv2/Conv2D_1/ReadVariableOp<model_14/model_13/vgg19/block2_conv2/Conv2D_1/ReadVariableOp2z
;model_14/model_13/vgg19/block3_conv1/BiasAdd/ReadVariableOp;model_14/model_13/vgg19/block3_conv1/BiasAdd/ReadVariableOp2~
=model_14/model_13/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp=model_14/model_13/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp2x
:model_14/model_13/vgg19/block3_conv1/Conv2D/ReadVariableOp:model_14/model_13/vgg19/block3_conv1/Conv2D/ReadVariableOp2|
<model_14/model_13/vgg19/block3_conv1/Conv2D_1/ReadVariableOp<model_14/model_13/vgg19/block3_conv1/Conv2D_1/ReadVariableOp2z
;model_14/model_13/vgg19/block3_conv2/BiasAdd/ReadVariableOp;model_14/model_13/vgg19/block3_conv2/BiasAdd/ReadVariableOp2~
=model_14/model_13/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp=model_14/model_13/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp2x
:model_14/model_13/vgg19/block3_conv2/Conv2D/ReadVariableOp:model_14/model_13/vgg19/block3_conv2/Conv2D/ReadVariableOp2|
<model_14/model_13/vgg19/block3_conv2/Conv2D_1/ReadVariableOp<model_14/model_13/vgg19/block3_conv2/Conv2D_1/ReadVariableOp2z
;model_14/model_13/vgg19/block3_conv3/BiasAdd/ReadVariableOp;model_14/model_13/vgg19/block3_conv3/BiasAdd/ReadVariableOp2~
=model_14/model_13/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp=model_14/model_13/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp2x
:model_14/model_13/vgg19/block3_conv3/Conv2D/ReadVariableOp:model_14/model_13/vgg19/block3_conv3/Conv2D/ReadVariableOp2|
<model_14/model_13/vgg19/block3_conv3/Conv2D_1/ReadVariableOp<model_14/model_13/vgg19/block3_conv3/Conv2D_1/ReadVariableOp2z
;model_14/model_13/vgg19/block3_conv4/BiasAdd/ReadVariableOp;model_14/model_13/vgg19/block3_conv4/BiasAdd/ReadVariableOp2~
=model_14/model_13/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp=model_14/model_13/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp2x
:model_14/model_13/vgg19/block3_conv4/Conv2D/ReadVariableOp:model_14/model_13/vgg19/block3_conv4/Conv2D/ReadVariableOp2|
<model_14/model_13/vgg19/block3_conv4/Conv2D_1/ReadVariableOp<model_14/model_13/vgg19/block3_conv4/Conv2D_1/ReadVariableOp2z
;model_14/model_13/vgg19/block4_conv1/BiasAdd/ReadVariableOp;model_14/model_13/vgg19/block4_conv1/BiasAdd/ReadVariableOp2~
=model_14/model_13/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp=model_14/model_13/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp2x
:model_14/model_13/vgg19/block4_conv1/Conv2D/ReadVariableOp:model_14/model_13/vgg19/block4_conv1/Conv2D/ReadVariableOp2|
<model_14/model_13/vgg19/block4_conv1/Conv2D_1/ReadVariableOp<model_14/model_13/vgg19/block4_conv1/Conv2D_1/ReadVariableOp2z
;model_14/model_13/vgg19/block4_conv2/BiasAdd/ReadVariableOp;model_14/model_13/vgg19/block4_conv2/BiasAdd/ReadVariableOp2~
=model_14/model_13/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp=model_14/model_13/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp2x
:model_14/model_13/vgg19/block4_conv2/Conv2D/ReadVariableOp:model_14/model_13/vgg19/block4_conv2/Conv2D/ReadVariableOp2|
<model_14/model_13/vgg19/block4_conv2/Conv2D_1/ReadVariableOp<model_14/model_13/vgg19/block4_conv2/Conv2D_1/ReadVariableOp2z
;model_14/model_13/vgg19/block4_conv3/BiasAdd/ReadVariableOp;model_14/model_13/vgg19/block4_conv3/BiasAdd/ReadVariableOp2~
=model_14/model_13/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp=model_14/model_13/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp2x
:model_14/model_13/vgg19/block4_conv3/Conv2D/ReadVariableOp:model_14/model_13/vgg19/block4_conv3/Conv2D/ReadVariableOp2|
<model_14/model_13/vgg19/block4_conv3/Conv2D_1/ReadVariableOp<model_14/model_13/vgg19/block4_conv3/Conv2D_1/ReadVariableOp2z
;model_14/model_13/vgg19/block4_conv4/BiasAdd/ReadVariableOp;model_14/model_13/vgg19/block4_conv4/BiasAdd/ReadVariableOp2~
=model_14/model_13/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp=model_14/model_13/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp2x
:model_14/model_13/vgg19/block4_conv4/Conv2D/ReadVariableOp:model_14/model_13/vgg19/block4_conv4/Conv2D/ReadVariableOp2|
<model_14/model_13/vgg19/block4_conv4/Conv2D_1/ReadVariableOp<model_14/model_13/vgg19/block4_conv4/Conv2D_1/ReadVariableOp2z
;model_14/model_13/vgg19/block5_conv1/BiasAdd/ReadVariableOp;model_14/model_13/vgg19/block5_conv1/BiasAdd/ReadVariableOp2~
=model_14/model_13/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp=model_14/model_13/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp2x
:model_14/model_13/vgg19/block5_conv1/Conv2D/ReadVariableOp:model_14/model_13/vgg19/block5_conv1/Conv2D/ReadVariableOp2|
<model_14/model_13/vgg19/block5_conv1/Conv2D_1/ReadVariableOp<model_14/model_13/vgg19/block5_conv1/Conv2D_1/ReadVariableOp2z
;model_14/model_13/vgg19/block5_conv2/BiasAdd/ReadVariableOp;model_14/model_13/vgg19/block5_conv2/BiasAdd/ReadVariableOp2~
=model_14/model_13/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp=model_14/model_13/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp2x
:model_14/model_13/vgg19/block5_conv2/Conv2D/ReadVariableOp:model_14/model_13/vgg19/block5_conv2/Conv2D/ReadVariableOp2|
<model_14/model_13/vgg19/block5_conv2/Conv2D_1/ReadVariableOp<model_14/model_13/vgg19/block5_conv2/Conv2D_1/ReadVariableOp2z
;model_14/model_13/vgg19/block5_conv3/BiasAdd/ReadVariableOp;model_14/model_13/vgg19/block5_conv3/BiasAdd/ReadVariableOp2~
=model_14/model_13/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp=model_14/model_13/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp2x
:model_14/model_13/vgg19/block5_conv3/Conv2D/ReadVariableOp:model_14/model_13/vgg19/block5_conv3/Conv2D/ReadVariableOp2|
<model_14/model_13/vgg19/block5_conv3/Conv2D_1/ReadVariableOp<model_14/model_13/vgg19/block5_conv3/Conv2D_1/ReadVariableOp2z
;model_14/model_13/vgg19/block5_conv4/BiasAdd/ReadVariableOp;model_14/model_13/vgg19/block5_conv4/BiasAdd/ReadVariableOp2~
=model_14/model_13/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp=model_14/model_13/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp2x
:model_14/model_13/vgg19/block5_conv4/Conv2D/ReadVariableOp:model_14/model_13/vgg19/block5_conv4/Conv2D/ReadVariableOp2|
<model_14/model_13/vgg19/block5_conv4/Conv2D_1/ReadVariableOp<model_14/model_13/vgg19/block5_conv4/Conv2D_1/ReadVariableOp:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
input_27:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
input_28
å
ÿ	
)__inference_model_14_layer_call_fn_977038
input_27
input_28!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	

unknown_31:	@

unknown_32:@

unknown_33:

unknown_34:
identity¢StatefulPartitionedCall¼
StatefulPartitionedCallStatefulPartitionedCallinput_27input_28unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_14_layer_call_and_return_conditional_losses_976885o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
input_27:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
input_28
£g

A__inference_vgg19_layer_call_and_return_conditional_losses_975101

inputs-
block1_conv1_974835:@!
block1_conv1_974837:@-
block1_conv2_974852:@@!
block1_conv2_974854:@.
block2_conv1_974870:@"
block2_conv1_974872:	/
block2_conv2_974887:"
block2_conv2_974889:	/
block3_conv1_974905:"
block3_conv1_974907:	/
block3_conv2_974922:"
block3_conv2_974924:	/
block3_conv3_974939:"
block3_conv3_974941:	/
block3_conv4_974956:"
block3_conv4_974958:	/
block4_conv1_974974:"
block4_conv1_974976:	/
block4_conv2_974991:"
block4_conv2_974993:	/
block4_conv3_975008:"
block4_conv3_975010:	/
block4_conv4_975025:"
block4_conv4_975027:	/
block5_conv1_975043:"
block5_conv1_975045:	/
block5_conv2_975060:"
block5_conv2_975062:	/
block5_conv3_975077:"
block5_conv3_975079:	/
block5_conv4_975094:"
block5_conv4_975096:	
identity¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block3_conv4/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall¢$block4_conv2/StatefulPartitionedCall¢$block4_conv3/StatefulPartitionedCall¢$block4_conv4/StatefulPartitionedCall¢$block5_conv1/StatefulPartitionedCall¢$block5_conv2/StatefulPartitionedCall¢$block5_conv3/StatefulPartitionedCall¢$block5_conv4/StatefulPartitionedCall
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_974835block1_conv1_974837*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block1_conv1_layer_call_and_return_conditional_losses_974834´
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_974852block1_conv2_974854*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block1_conv2_layer_call_and_return_conditional_losses_974851ð
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block1_pool_layer_call_and_return_conditional_losses_974765ª
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_974870block2_conv1_974872*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block2_conv1_layer_call_and_return_conditional_losses_974869³
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_974887block2_conv2_974889*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block2_conv2_layer_call_and_return_conditional_losses_974886ñ
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block2_pool_layer_call_and_return_conditional_losses_974777ª
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_974905block3_conv1_974907*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv1_layer_call_and_return_conditional_losses_974904³
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_974922block3_conv2_974924*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv2_layer_call_and_return_conditional_losses_974921³
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_974939block3_conv3_974941*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv3_layer_call_and_return_conditional_losses_974938³
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_974956block3_conv4_974958*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv4_layer_call_and_return_conditional_losses_974955ñ
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_pool_layer_call_and_return_conditional_losses_974789ª
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_974974block4_conv1_974976*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv1_layer_call_and_return_conditional_losses_974973³
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_974991block4_conv2_974993*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv2_layer_call_and_return_conditional_losses_974990³
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_975008block4_conv3_975010*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv3_layer_call_and_return_conditional_losses_975007³
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_975025block4_conv4_975027*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv4_layer_call_and_return_conditional_losses_975024ñ
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_pool_layer_call_and_return_conditional_losses_974801ª
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_975043block5_conv1_975045*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv1_layer_call_and_return_conditional_losses_975042³
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_975060block5_conv2_975062*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv2_layer_call_and_return_conditional_losses_975059³
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_975077block5_conv3_975079*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv3_layer_call_and_return_conditional_losses_975076³
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_975094block5_conv4_975096*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv4_layer_call_and_return_conditional_losses_975093ñ
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_pool_layer_call_and_return_conditional_losses_974813|
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs
³
H
,__inference_block1_pool_layer_call_fn_978901

inputs
identityØ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block1_pool_layer_call_and_return_conditional_losses_974765
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
H
,__inference_block3_pool_layer_call_fn_979041

inputs
identityØ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_pool_layer_call_and_return_conditional_losses_974789
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤
é
A__inference_vgg19_layer_call_and_return_conditional_losses_978826

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@;
,block2_conv1_biasadd_readvariableop_resource:	G
+block2_conv2_conv2d_readvariableop_resource:;
,block2_conv2_biasadd_readvariableop_resource:	G
+block3_conv1_conv2d_readvariableop_resource:;
,block3_conv1_biasadd_readvariableop_resource:	G
+block3_conv2_conv2d_readvariableop_resource:;
,block3_conv2_biasadd_readvariableop_resource:	G
+block3_conv3_conv2d_readvariableop_resource:;
,block3_conv3_biasadd_readvariableop_resource:	G
+block3_conv4_conv2d_readvariableop_resource:;
,block3_conv4_biasadd_readvariableop_resource:	G
+block4_conv1_conv2d_readvariableop_resource:;
,block4_conv1_biasadd_readvariableop_resource:	G
+block4_conv2_conv2d_readvariableop_resource:;
,block4_conv2_biasadd_readvariableop_resource:	G
+block4_conv3_conv2d_readvariableop_resource:;
,block4_conv3_biasadd_readvariableop_resource:	G
+block4_conv4_conv2d_readvariableop_resource:;
,block4_conv4_biasadd_readvariableop_resource:	G
+block5_conv1_conv2d_readvariableop_resource:;
,block5_conv1_biasadd_readvariableop_resource:	G
+block5_conv2_conv2d_readvariableop_resource:;
,block5_conv2_biasadd_readvariableop_resource:	G
+block5_conv3_conv2d_readvariableop_resource:;
,block5_conv3_biasadd_readvariableop_resource:	G
+block5_conv4_conv2d_readvariableop_resource:;
,block5_conv4_biasadd_readvariableop_resource:	
identity¢#block1_conv1/BiasAdd/ReadVariableOp¢"block1_conv1/Conv2D/ReadVariableOp¢#block1_conv2/BiasAdd/ReadVariableOp¢"block1_conv2/Conv2D/ReadVariableOp¢#block2_conv1/BiasAdd/ReadVariableOp¢"block2_conv1/Conv2D/ReadVariableOp¢#block2_conv2/BiasAdd/ReadVariableOp¢"block2_conv2/Conv2D/ReadVariableOp¢#block3_conv1/BiasAdd/ReadVariableOp¢"block3_conv1/Conv2D/ReadVariableOp¢#block3_conv2/BiasAdd/ReadVariableOp¢"block3_conv2/Conv2D/ReadVariableOp¢#block3_conv3/BiasAdd/ReadVariableOp¢"block3_conv3/Conv2D/ReadVariableOp¢#block3_conv4/BiasAdd/ReadVariableOp¢"block3_conv4/Conv2D/ReadVariableOp¢#block4_conv1/BiasAdd/ReadVariableOp¢"block4_conv1/Conv2D/ReadVariableOp¢#block4_conv2/BiasAdd/ReadVariableOp¢"block4_conv2/Conv2D/ReadVariableOp¢#block4_conv3/BiasAdd/ReadVariableOp¢"block4_conv3/Conv2D/ReadVariableOp¢#block4_conv4/BiasAdd/ReadVariableOp¢"block4_conv4/Conv2D/ReadVariableOp¢#block5_conv1/BiasAdd/ReadVariableOp¢"block5_conv1/Conv2D/ReadVariableOp¢#block5_conv2/BiasAdd/ReadVariableOp¢"block5_conv2/Conv2D/ReadVariableOp¢#block5_conv3/BiasAdd/ReadVariableOp¢"block5_conv3/Conv2D/ReadVariableOp¢#block5_conv4/BiasAdd/ReadVariableOp¢"block5_conv4/Conv2D/ReadVariableOp
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0µ
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides

#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Î
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides

#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@¬
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@*
ksize
*
paddingVALID*
strides

"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ê
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides

#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdds
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides

#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdds
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd­
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
ksize
*
paddingVALID*
strides

"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22s
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22s
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22s
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22s
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22­
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block4_conv4/Conv2D/ReadVariableOpReadVariableOp+block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block4_conv4/Conv2DConv2Dblock4_conv3/Relu:activations:0*block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv4/BiasAdd/ReadVariableOpReadVariableOp,block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv4/BiasAddBiasAddblock4_conv4/Conv2D:output:0+block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv4/ReluRelublock4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
block4_pool/MaxPoolMaxPoolblock4_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block5_conv4/Conv2D/ReadVariableOpReadVariableOp+block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block5_conv4/Conv2DConv2Dblock5_conv3/Relu:activations:0*block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv4/BiasAdd/ReadVariableOpReadVariableOp,block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv4/BiasAddBiasAddblock5_conv4/Conv2D:output:0+block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv4/ReluRelublock5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
block5_pool/MaxPoolMaxPoolblock5_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
t
IdentityIdentityblock5_pool/MaxPool:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö	
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block3_conv4/BiasAdd/ReadVariableOp#^block3_conv4/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block4_conv4/BiasAdd/ReadVariableOp#^block4_conv4/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp$^block5_conv4/BiasAdd/ReadVariableOp#^block5_conv4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block3_conv4/BiasAdd/ReadVariableOp#block3_conv4/BiasAdd/ReadVariableOp2H
"block3_conv4/Conv2D/ReadVariableOp"block3_conv4/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block4_conv4/BiasAdd/ReadVariableOp#block4_conv4/BiasAdd/ReadVariableOp2H
"block4_conv4/Conv2D/ReadVariableOp"block4_conv4/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp2J
#block5_conv4/BiasAdd/ReadVariableOp#block5_conv4/BiasAdd/ReadVariableOp2H
"block5_conv4/Conv2D/ReadVariableOp"block5_conv4/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs


H__inference_block3_conv3_layer_call_and_return_conditional_losses_974938

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs


H__inference_block5_conv2_layer_call_and_return_conditional_losses_975059

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
ý
&__inference_vgg19_layer_call_fn_975168
input_30!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinput_30unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_vgg19_layer_call_and_return_conditional_losses_975101x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
input_30


H__inference_block5_conv2_layer_call_and_return_conditional_losses_979176

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
H
,__inference_block4_pool_layer_call_fn_979131

inputs
identityØ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_pool_layer_call_and_return_conditional_losses_974801
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤
û
&__inference_vgg19_layer_call_fn_978515

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	
identity¢StatefulPartitionedCallý
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
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_vgg19_layer_call_and_return_conditional_losses_975101x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs
¥
U
)__inference_lambda_4_layer_call_fn_978392
inputs_0
inputs_1
identity¿
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_lambda_4_layer_call_and_return_conditional_losses_976557`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/1


H__inference_block1_conv2_layer_call_and_return_conditional_losses_974851

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÈÈ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
 
_user_specified_nameinputs
å
ÿ	
)__inference_model_14_layer_call_fn_976652
input_27
input_28!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	

unknown_31:	@

unknown_32:@

unknown_33:

unknown_34:
identity¢StatefulPartitionedCall¼
StatefulPartitionedCallStatefulPartitionedCallinput_27input_28unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_14_layer_call_and_return_conditional_losses_976577o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
input_27:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
input_28


H__inference_block1_conv1_layer_call_and_return_conditional_losses_974834

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÈÈ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs


õ
D__inference_dense_14_layer_call_and_return_conditional_losses_976570

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·
r
V__inference_global_average_pooling2d_9_layer_call_and_return_conditional_losses_978837

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
¥
-__inference_block4_conv1_layer_call_fn_979055

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv1_layer_call_and_return_conditional_losses_974973x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block2_conv1_layer_call_and_return_conditional_losses_974869

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿdd@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@
 
_user_specified_nameinputs


H__inference_block3_conv1_layer_call_and_return_conditional_losses_974904

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
¿
£
D__inference_model_13_layer_call_and_return_conditional_losses_978386

inputsK
1vgg19_block1_conv1_conv2d_readvariableop_resource:@@
2vgg19_block1_conv1_biasadd_readvariableop_resource:@K
1vgg19_block1_conv2_conv2d_readvariableop_resource:@@@
2vgg19_block1_conv2_biasadd_readvariableop_resource:@L
1vgg19_block2_conv1_conv2d_readvariableop_resource:@A
2vgg19_block2_conv1_biasadd_readvariableop_resource:	M
1vgg19_block2_conv2_conv2d_readvariableop_resource:A
2vgg19_block2_conv2_biasadd_readvariableop_resource:	M
1vgg19_block3_conv1_conv2d_readvariableop_resource:A
2vgg19_block3_conv1_biasadd_readvariableop_resource:	M
1vgg19_block3_conv2_conv2d_readvariableop_resource:A
2vgg19_block3_conv2_biasadd_readvariableop_resource:	M
1vgg19_block3_conv3_conv2d_readvariableop_resource:A
2vgg19_block3_conv3_biasadd_readvariableop_resource:	M
1vgg19_block3_conv4_conv2d_readvariableop_resource:A
2vgg19_block3_conv4_biasadd_readvariableop_resource:	M
1vgg19_block4_conv1_conv2d_readvariableop_resource:A
2vgg19_block4_conv1_biasadd_readvariableop_resource:	M
1vgg19_block4_conv2_conv2d_readvariableop_resource:A
2vgg19_block4_conv2_biasadd_readvariableop_resource:	M
1vgg19_block4_conv3_conv2d_readvariableop_resource:A
2vgg19_block4_conv3_biasadd_readvariableop_resource:	M
1vgg19_block4_conv4_conv2d_readvariableop_resource:A
2vgg19_block4_conv4_biasadd_readvariableop_resource:	M
1vgg19_block5_conv1_conv2d_readvariableop_resource:A
2vgg19_block5_conv1_biasadd_readvariableop_resource:	M
1vgg19_block5_conv2_conv2d_readvariableop_resource:A
2vgg19_block5_conv2_biasadd_readvariableop_resource:	M
1vgg19_block5_conv3_conv2d_readvariableop_resource:A
2vgg19_block5_conv3_biasadd_readvariableop_resource:	M
1vgg19_block5_conv4_conv2d_readvariableop_resource:A
2vgg19_block5_conv4_biasadd_readvariableop_resource:	:
'dense_13_matmul_readvariableop_resource:	@6
(dense_13_biasadd_readvariableop_resource:@
identity¢dense_13/BiasAdd/ReadVariableOp¢dense_13/MatMul/ReadVariableOp¢)vgg19/block1_conv1/BiasAdd/ReadVariableOp¢(vgg19/block1_conv1/Conv2D/ReadVariableOp¢)vgg19/block1_conv2/BiasAdd/ReadVariableOp¢(vgg19/block1_conv2/Conv2D/ReadVariableOp¢)vgg19/block2_conv1/BiasAdd/ReadVariableOp¢(vgg19/block2_conv1/Conv2D/ReadVariableOp¢)vgg19/block2_conv2/BiasAdd/ReadVariableOp¢(vgg19/block2_conv2/Conv2D/ReadVariableOp¢)vgg19/block3_conv1/BiasAdd/ReadVariableOp¢(vgg19/block3_conv1/Conv2D/ReadVariableOp¢)vgg19/block3_conv2/BiasAdd/ReadVariableOp¢(vgg19/block3_conv2/Conv2D/ReadVariableOp¢)vgg19/block3_conv3/BiasAdd/ReadVariableOp¢(vgg19/block3_conv3/Conv2D/ReadVariableOp¢)vgg19/block3_conv4/BiasAdd/ReadVariableOp¢(vgg19/block3_conv4/Conv2D/ReadVariableOp¢)vgg19/block4_conv1/BiasAdd/ReadVariableOp¢(vgg19/block4_conv1/Conv2D/ReadVariableOp¢)vgg19/block4_conv2/BiasAdd/ReadVariableOp¢(vgg19/block4_conv2/Conv2D/ReadVariableOp¢)vgg19/block4_conv3/BiasAdd/ReadVariableOp¢(vgg19/block4_conv3/Conv2D/ReadVariableOp¢)vgg19/block4_conv4/BiasAdd/ReadVariableOp¢(vgg19/block4_conv4/Conv2D/ReadVariableOp¢)vgg19/block5_conv1/BiasAdd/ReadVariableOp¢(vgg19/block5_conv1/Conv2D/ReadVariableOp¢)vgg19/block5_conv2/BiasAdd/ReadVariableOp¢(vgg19/block5_conv2/Conv2D/ReadVariableOp¢)vgg19/block5_conv3/BiasAdd/ReadVariableOp¢(vgg19/block5_conv3/Conv2D/ReadVariableOp¢)vgg19/block5_conv4/BiasAdd/ReadVariableOp¢(vgg19/block5_conv4/Conv2D/ReadVariableOp¢
(vgg19/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Á
vgg19/block1_conv1/Conv2DConv2Dinputs0vgg19/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides

)vgg19/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¸
vgg19/block1_conv1/BiasAddBiasAdd"vgg19/block1_conv1/Conv2D:output:01vgg19/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
vgg19/block1_conv1/ReluRelu#vgg19/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@¢
(vgg19/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0à
vgg19/block1_conv2/Conv2DConv2D%vgg19/block1_conv1/Relu:activations:00vgg19/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides

)vgg19/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¸
vgg19/block1_conv2/BiasAddBiasAdd"vgg19/block1_conv2/Conv2D:output:01vgg19/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
vgg19/block1_conv2/ReluRelu#vgg19/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@¸
vgg19/block1_pool/MaxPoolMaxPool%vgg19/block1_conv2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@*
ksize
*
paddingVALID*
strides
£
(vgg19/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ü
vgg19/block2_conv1/Conv2DConv2D"vgg19/block1_pool/MaxPool:output:00vgg19/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides

)vgg19/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block2_conv1/BiasAddBiasAdd"vgg19/block2_conv1/Conv2D:output:01vgg19/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
vgg19/block2_conv1/ReluRelu#vgg19/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd¤
(vgg19/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block2_conv2/Conv2DConv2D%vgg19/block2_conv1/Relu:activations:00vgg19/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides

)vgg19/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block2_conv2/BiasAddBiasAdd"vgg19/block2_conv2/Conv2D:output:01vgg19/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
vgg19/block2_conv2/ReluRelu#vgg19/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd¹
vgg19/block2_pool/MaxPoolMaxPool%vgg19/block2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
ksize
*
paddingVALID*
strides
¤
(vgg19/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
vgg19/block3_conv1/Conv2DConv2D"vgg19/block2_pool/MaxPool:output:00vgg19/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

)vgg19/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block3_conv1/BiasAddBiasAdd"vgg19/block3_conv1/Conv2D:output:01vgg19/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
vgg19/block3_conv1/ReluRelu#vgg19/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¤
(vgg19/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block3_conv2/Conv2DConv2D%vgg19/block3_conv1/Relu:activations:00vgg19/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

)vgg19/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block3_conv2/BiasAddBiasAdd"vgg19/block3_conv2/Conv2D:output:01vgg19/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
vgg19/block3_conv2/ReluRelu#vgg19/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¤
(vgg19/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block3_conv3/Conv2DConv2D%vgg19/block3_conv2/Relu:activations:00vgg19/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

)vgg19/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block3_conv3/BiasAddBiasAdd"vgg19/block3_conv3/Conv2D:output:01vgg19/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
vgg19/block3_conv3/ReluRelu#vgg19/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¤
(vgg19/block3_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block3_conv4/Conv2DConv2D%vgg19/block3_conv3/Relu:activations:00vgg19/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

)vgg19/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block3_conv4/BiasAddBiasAdd"vgg19/block3_conv4/Conv2D:output:01vgg19/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
vgg19/block3_conv4/ReluRelu#vgg19/block3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¹
vgg19/block3_pool/MaxPoolMaxPool%vgg19/block3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¤
(vgg19/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
vgg19/block4_conv1/Conv2DConv2D"vgg19/block3_pool/MaxPool:output:00vgg19/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block4_conv1/BiasAddBiasAdd"vgg19/block4_conv1/Conv2D:output:01vgg19/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block4_conv1/ReluRelu#vgg19/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg19/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block4_conv2/Conv2DConv2D%vgg19/block4_conv1/Relu:activations:00vgg19/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block4_conv2/BiasAddBiasAdd"vgg19/block4_conv2/Conv2D:output:01vgg19/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block4_conv2/ReluRelu#vgg19/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg19/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block4_conv3/Conv2DConv2D%vgg19/block4_conv2/Relu:activations:00vgg19/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block4_conv3/BiasAddBiasAdd"vgg19/block4_conv3/Conv2D:output:01vgg19/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block4_conv3/ReluRelu#vgg19/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg19/block4_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block4_conv4/Conv2DConv2D%vgg19/block4_conv3/Relu:activations:00vgg19/block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block4_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block4_conv4/BiasAddBiasAdd"vgg19/block4_conv4/Conv2D:output:01vgg19/block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block4_conv4/ReluRelu#vgg19/block4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
vgg19/block4_pool/MaxPoolMaxPool%vgg19/block4_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¤
(vgg19/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
vgg19/block5_conv1/Conv2DConv2D"vgg19/block4_pool/MaxPool:output:00vgg19/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block5_conv1/BiasAddBiasAdd"vgg19/block5_conv1/Conv2D:output:01vgg19/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block5_conv1/ReluRelu#vgg19/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg19/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block5_conv2/Conv2DConv2D%vgg19/block5_conv1/Relu:activations:00vgg19/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block5_conv2/BiasAddBiasAdd"vgg19/block5_conv2/Conv2D:output:01vgg19/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block5_conv2/ReluRelu#vgg19/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg19/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block5_conv3/Conv2DConv2D%vgg19/block5_conv2/Relu:activations:00vgg19/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block5_conv3/BiasAddBiasAdd"vgg19/block5_conv3/Conv2D:output:01vgg19/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block5_conv3/ReluRelu#vgg19/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg19/block5_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block5_conv4/Conv2DConv2D%vgg19/block5_conv3/Relu:activations:00vgg19/block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block5_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block5_conv4/BiasAddBiasAdd"vgg19/block5_conv4/Conv2D:output:01vgg19/block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block5_conv4/ReluRelu#vgg19/block5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
vgg19/block5_pool/MaxPoolMaxPool%vgg19/block5_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

1global_average_pooling2d_9/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      º
global_average_pooling2d_9/MeanMean"vgg19/block5_pool/MaxPool:output:0:global_average_pooling2d_9/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
dense_13/MatMulMatMul(global_average_pooling2d_9/Mean:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
IdentityIdentitydense_13/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ù
NoOpNoOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*^vgg19/block1_conv1/BiasAdd/ReadVariableOp)^vgg19/block1_conv1/Conv2D/ReadVariableOp*^vgg19/block1_conv2/BiasAdd/ReadVariableOp)^vgg19/block1_conv2/Conv2D/ReadVariableOp*^vgg19/block2_conv1/BiasAdd/ReadVariableOp)^vgg19/block2_conv1/Conv2D/ReadVariableOp*^vgg19/block2_conv2/BiasAdd/ReadVariableOp)^vgg19/block2_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv1/BiasAdd/ReadVariableOp)^vgg19/block3_conv1/Conv2D/ReadVariableOp*^vgg19/block3_conv2/BiasAdd/ReadVariableOp)^vgg19/block3_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv3/BiasAdd/ReadVariableOp)^vgg19/block3_conv3/Conv2D/ReadVariableOp*^vgg19/block3_conv4/BiasAdd/ReadVariableOp)^vgg19/block3_conv4/Conv2D/ReadVariableOp*^vgg19/block4_conv1/BiasAdd/ReadVariableOp)^vgg19/block4_conv1/Conv2D/ReadVariableOp*^vgg19/block4_conv2/BiasAdd/ReadVariableOp)^vgg19/block4_conv2/Conv2D/ReadVariableOp*^vgg19/block4_conv3/BiasAdd/ReadVariableOp)^vgg19/block4_conv3/Conv2D/ReadVariableOp*^vgg19/block4_conv4/BiasAdd/ReadVariableOp)^vgg19/block4_conv4/Conv2D/ReadVariableOp*^vgg19/block5_conv1/BiasAdd/ReadVariableOp)^vgg19/block5_conv1/Conv2D/ReadVariableOp*^vgg19/block5_conv2/BiasAdd/ReadVariableOp)^vgg19/block5_conv2/Conv2D/ReadVariableOp*^vgg19/block5_conv3/BiasAdd/ReadVariableOp)^vgg19/block5_conv3/Conv2D/ReadVariableOp*^vgg19/block5_conv4/BiasAdd/ReadVariableOp)^vgg19/block5_conv4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2V
)vgg19/block1_conv1/BiasAdd/ReadVariableOp)vgg19/block1_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block1_conv1/Conv2D/ReadVariableOp(vgg19/block1_conv1/Conv2D/ReadVariableOp2V
)vgg19/block1_conv2/BiasAdd/ReadVariableOp)vgg19/block1_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block1_conv2/Conv2D/ReadVariableOp(vgg19/block1_conv2/Conv2D/ReadVariableOp2V
)vgg19/block2_conv1/BiasAdd/ReadVariableOp)vgg19/block2_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block2_conv1/Conv2D/ReadVariableOp(vgg19/block2_conv1/Conv2D/ReadVariableOp2V
)vgg19/block2_conv2/BiasAdd/ReadVariableOp)vgg19/block2_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block2_conv2/Conv2D/ReadVariableOp(vgg19/block2_conv2/Conv2D/ReadVariableOp2V
)vgg19/block3_conv1/BiasAdd/ReadVariableOp)vgg19/block3_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv1/Conv2D/ReadVariableOp(vgg19/block3_conv1/Conv2D/ReadVariableOp2V
)vgg19/block3_conv2/BiasAdd/ReadVariableOp)vgg19/block3_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv2/Conv2D/ReadVariableOp(vgg19/block3_conv2/Conv2D/ReadVariableOp2V
)vgg19/block3_conv3/BiasAdd/ReadVariableOp)vgg19/block3_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv3/Conv2D/ReadVariableOp(vgg19/block3_conv3/Conv2D/ReadVariableOp2V
)vgg19/block3_conv4/BiasAdd/ReadVariableOp)vgg19/block3_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv4/Conv2D/ReadVariableOp(vgg19/block3_conv4/Conv2D/ReadVariableOp2V
)vgg19/block4_conv1/BiasAdd/ReadVariableOp)vgg19/block4_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv1/Conv2D/ReadVariableOp(vgg19/block4_conv1/Conv2D/ReadVariableOp2V
)vgg19/block4_conv2/BiasAdd/ReadVariableOp)vgg19/block4_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv2/Conv2D/ReadVariableOp(vgg19/block4_conv2/Conv2D/ReadVariableOp2V
)vgg19/block4_conv3/BiasAdd/ReadVariableOp)vgg19/block4_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv3/Conv2D/ReadVariableOp(vgg19/block4_conv3/Conv2D/ReadVariableOp2V
)vgg19/block4_conv4/BiasAdd/ReadVariableOp)vgg19/block4_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv4/Conv2D/ReadVariableOp(vgg19/block4_conv4/Conv2D/ReadVariableOp2V
)vgg19/block5_conv1/BiasAdd/ReadVariableOp)vgg19/block5_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv1/Conv2D/ReadVariableOp(vgg19/block5_conv1/Conv2D/ReadVariableOp2V
)vgg19/block5_conv2/BiasAdd/ReadVariableOp)vgg19/block5_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv2/Conv2D/ReadVariableOp(vgg19/block5_conv2/Conv2D/ReadVariableOp2V
)vgg19/block5_conv3/BiasAdd/ReadVariableOp)vgg19/block5_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv3/Conv2D/ReadVariableOp(vgg19/block5_conv3/Conv2D/ReadVariableOp2V
)vgg19/block5_conv4/BiasAdd/ReadVariableOp)vgg19/block5_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv4/Conv2D/ReadVariableOp(vgg19/block5_conv4/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs
·
r
V__inference_global_average_pooling2d_9_layer_call_and_return_conditional_losses_975812

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block4_conv1_layer_call_and_return_conditional_losses_979066

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block1_conv2_layer_call_and_return_conditional_losses_978896

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÈÈ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
 
_user_specified_nameinputs


H__inference_block2_conv1_layer_call_and_return_conditional_losses_978926

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿdd@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@
 
_user_specified_nameinputs
ù
¤
-__inference_block2_conv1_layer_call_fn_978915

inputs"
unknown:@
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block2_conv1_layer_call_and_return_conditional_losses_974869x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿdd@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@
 
_user_specified_nameinputs
Å

)__inference_dense_14_layer_call_fn_978435

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_976570o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_defaultì
G
input_27;
serving_default_input_27:0ÿÿÿÿÿÿÿÿÿÈÈ
G
input_28;
serving_default_input_28:0ÿÿÿÿÿÿÿÿÿÈÈ<
dense_140
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:æÀ
Ø
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
	optimizer
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
"
_tf_keras_input_layer

layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_network
¥
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer

'iter

(beta_1

)beta_2
	*decaym mKmLmv vKvLv"
	optimizer
¶
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15
;16
<17
=18
>19
?20
@21
A22
B23
C24
D25
E26
F27
G28
H29
I30
J31
K32
L33
34
 35"
trackable_list_wrapper
<
K0
L1
2
 3"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ò2ï
)__inference_model_14_layer_call_fn_976652
)__inference_model_14_layer_call_fn_977350
)__inference_model_14_layer_call_fn_977428
)__inference_model_14_layer_call_fn_977038À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
D__inference_model_14_layer_call_and_return_conditional_losses_977665
D__inference_model_14_layer_call_and_return_conditional_losses_977902
D__inference_model_14_layer_call_and_return_conditional_losses_977153
D__inference_model_14_layer_call_and_return_conditional_losses_977268À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
×BÔ
!__inference__wrapped_model_974756input_27input_28"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
Rserving_default"
signature_map
"
_tf_keras_input_layer
÷
Slayer-0
Tlayer_with_weights-0
Tlayer-1
Ulayer_with_weights-1
Ulayer-2
Vlayer-3
Wlayer_with_weights-2
Wlayer-4
Xlayer_with_weights-3
Xlayer-5
Ylayer-6
Zlayer_with_weights-4
Zlayer-7
[layer_with_weights-5
[layer-8
\layer_with_weights-6
\layer-9
]layer_with_weights-7
]layer-10
^layer-11
_layer_with_weights-8
_layer-12
`layer_with_weights-9
`layer-13
alayer_with_weights-10
alayer-14
blayer_with_weights-11
blayer-15
clayer-16
dlayer_with_weights-12
dlayer-17
elayer_with_weights-13
elayer-18
flayer_with_weights-14
flayer-19
glayer_with_weights-15
glayer-20
hlayer-21
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_network
¥
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Kkernel
Lbias
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses"
_tf_keras_layer
¦
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15
;16
<17
=18
>19
?20
@21
A22
B23
C24
D25
E26
F27
G28
H29
I30
J31
K32
L33"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ò2ï
)__inference_model_13_layer_call_fn_975976
)__inference_model_13_layer_call_fn_978055
)__inference_model_13_layer_call_fn_978128
)__inference_model_13_layer_call_fn_976280À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
D__inference_model_13_layer_call_and_return_conditional_losses_978257
D__inference_model_13_layer_call_and_return_conditional_losses_978386
D__inference_model_13_layer_call_and_return_conditional_losses_976355
D__inference_model_13_layer_call_and_return_conditional_losses_976430À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2
)__inference_lambda_4_layer_call_fn_978392
)__inference_lambda_4_layer_call_fn_978398À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
D__inference_lambda_4_layer_call_and_return_conditional_losses_978412
D__inference_lambda_4_layer_call_and_return_conditional_losses_978426À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
!:2dense_14/kernel
:2dense_14/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_14_layer_call_fn_978435¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_14_layer_call_and_return_conditional_losses_978446¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
-:+@2block1_conv1/kernel
:@2block1_conv1/bias
-:+@@2block1_conv2/kernel
:@2block1_conv2/bias
.:,@2block2_conv1/kernel
 :2block2_conv1/bias
/:-2block2_conv2/kernel
 :2block2_conv2/bias
/:-2block3_conv1/kernel
 :2block3_conv1/bias
/:-2block3_conv2/kernel
 :2block3_conv2/bias
/:-2block3_conv3/kernel
 :2block3_conv3/bias
/:-2block3_conv4/kernel
 :2block3_conv4/bias
/:-2block4_conv1/kernel
 :2block4_conv1/bias
/:-2block4_conv2/kernel
 :2block4_conv2/bias
/:-2block4_conv3/kernel
 :2block4_conv3/bias
/:-2block4_conv4/kernel
 :2block4_conv4/bias
/:-2block5_conv1/kernel
 :2block5_conv1/bias
/:-2block5_conv2/kernel
 :2block5_conv2/bias
/:-2block5_conv3/kernel
 :2block5_conv3/bias
/:-2block5_conv4/kernel
 :2block5_conv4/bias
": 	@2dense_13/kernel
:@2dense_13/bias

+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15
;16
<17
=18
>19
?20
@21
A22
B23
C24
D25
E26
F27
G28
H29
I30
J31"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÔBÑ
$__inference_signature_wrapper_977982input_27input_28"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
"
_tf_keras_input_layer
Á

+kernel
,bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

-kernel
.bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

/kernel
0bias
	variables
trainable_variables
 regularization_losses
¡	keras_api
¢__call__
+£&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

1kernel
2bias
¤	variables
¥trainable_variables
¦regularization_losses
§	keras_api
¨__call__
+©&call_and_return_all_conditional_losses"
_tf_keras_layer
«
ª	variables
«trainable_variables
¬regularization_losses
­	keras_api
®__call__
+¯&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

3kernel
4bias
°	variables
±trainable_variables
²regularization_losses
³	keras_api
´__call__
+µ&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

5kernel
6bias
¶	variables
·trainable_variables
¸regularization_losses
¹	keras_api
º__call__
+»&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

7kernel
8bias
¼	variables
½trainable_variables
¾regularization_losses
¿	keras_api
À__call__
+Á&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

9kernel
:bias
Â	variables
Ãtrainable_variables
Äregularization_losses
Å	keras_api
Æ__call__
+Ç&call_and_return_all_conditional_losses"
_tf_keras_layer
«
È	variables
Étrainable_variables
Êregularization_losses
Ë	keras_api
Ì__call__
+Í&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

;kernel
<bias
Î	variables
Ïtrainable_variables
Ðregularization_losses
Ñ	keras_api
Ò__call__
+Ó&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

=kernel
>bias
Ô	variables
Õtrainable_variables
Öregularization_losses
×	keras_api
Ø__call__
+Ù&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

?kernel
@bias
Ú	variables
Ûtrainable_variables
Üregularization_losses
Ý	keras_api
Þ__call__
+ß&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

Akernel
Bbias
à	variables
átrainable_variables
âregularization_losses
ã	keras_api
ä__call__
+å&call_and_return_all_conditional_losses"
_tf_keras_layer
«
æ	variables
çtrainable_variables
èregularization_losses
é	keras_api
ê__call__
+ë&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

Ckernel
Dbias
ì	variables
ítrainable_variables
îregularization_losses
ï	keras_api
ð__call__
+ñ&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

Ekernel
Fbias
ò	variables
ótrainable_variables
ôregularization_losses
õ	keras_api
ö__call__
+÷&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

Gkernel
Hbias
ø	variables
ùtrainable_variables
úregularization_losses
û	keras_api
ü__call__
+ý&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

Ikernel
Jbias
þ	variables
ÿtrainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer

+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15
;16
<17
=18
>19
?20
@21
A22
B23
C24
D25
E26
F27
G28
H29
I30
J31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
æ2ã
&__inference_vgg19_layer_call_fn_975168
&__inference_vgg19_layer_call_fn_978515
&__inference_vgg19_layer_call_fn_978584
&__inference_vgg19_layer_call_fn_975624À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
A__inference_vgg19_layer_call_and_return_conditional_losses_978705
A__inference_vgg19_layer_call_and_return_conditional_losses_978826
A__inference_vgg19_layer_call_and_return_conditional_losses_975713
A__inference_vgg19_layer_call_and_return_conditional_losses_975802À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
å2â
;__inference_global_average_pooling2d_9_layer_call_fn_978831¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2ý
V__inference_global_average_pooling2d_9_layer_call_and_return_conditional_losses_978837¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_13_layer_call_fn_978846¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_13_layer_call_and_return_conditional_losses_978856¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 

+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15
;16
<17
=18
>19
?20
@21
A22
B23
C24
D25
E26
F27
G28
H29
I30
J31"
trackable_list_wrapper
<
0
1
2
3"
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
R

total

count
	variables
	keras_api"
_tf_keras_metric
c

total

count

_fn_kwargs
 	variables
¡	keras_api"
_tf_keras_metric
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block1_conv1_layer_call_fn_978865¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block1_conv1_layer_call_and_return_conditional_losses_978876¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block1_conv2_layer_call_fn_978885¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block1_conv2_layer_call_and_return_conditional_losses_978896¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_block1_pool_layer_call_fn_978901¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_block1_pool_layer_call_and_return_conditional_losses_978906¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
	variables
trainable_variables
 regularization_losses
¢__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block2_conv1_layer_call_fn_978915¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block2_conv1_layer_call_and_return_conditional_losses_978926¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
¤	variables
¥trainable_variables
¦regularization_losses
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block2_conv2_layer_call_fn_978935¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block2_conv2_layer_call_and_return_conditional_losses_978946¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
»non_trainable_variables
¼layers
½metrics
 ¾layer_regularization_losses
¿layer_metrics
ª	variables
«trainable_variables
¬regularization_losses
®__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_block2_pool_layer_call_fn_978951¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_block2_pool_layer_call_and_return_conditional_losses_978956¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ànon_trainable_variables
Álayers
Âmetrics
 Ãlayer_regularization_losses
Älayer_metrics
°	variables
±trainable_variables
²regularization_losses
´__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block3_conv1_layer_call_fn_978965¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block3_conv1_layer_call_and_return_conditional_losses_978976¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ånon_trainable_variables
Ælayers
Çmetrics
 Èlayer_regularization_losses
Élayer_metrics
¶	variables
·trainable_variables
¸regularization_losses
º__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block3_conv2_layer_call_fn_978985¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block3_conv2_layer_call_and_return_conditional_losses_978996¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ênon_trainable_variables
Ëlayers
Ìmetrics
 Ílayer_regularization_losses
Îlayer_metrics
¼	variables
½trainable_variables
¾regularization_losses
À__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block3_conv3_layer_call_fn_979005¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block3_conv3_layer_call_and_return_conditional_losses_979016¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ïnon_trainable_variables
Ðlayers
Ñmetrics
 Òlayer_regularization_losses
Ólayer_metrics
Â	variables
Ãtrainable_variables
Äregularization_losses
Æ__call__
+Ç&call_and_return_all_conditional_losses
'Ç"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block3_conv4_layer_call_fn_979025¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block3_conv4_layer_call_and_return_conditional_losses_979036¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ônon_trainable_variables
Õlayers
Ömetrics
 ×layer_regularization_losses
Ølayer_metrics
È	variables
Étrainable_variables
Êregularization_losses
Ì__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_block3_pool_layer_call_fn_979041¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_block3_pool_layer_call_and_return_conditional_losses_979046¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ùnon_trainable_variables
Úlayers
Ûmetrics
 Ülayer_regularization_losses
Ýlayer_metrics
Î	variables
Ïtrainable_variables
Ðregularization_losses
Ò__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block4_conv1_layer_call_fn_979055¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block4_conv1_layer_call_and_return_conditional_losses_979066¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Þnon_trainable_variables
ßlayers
àmetrics
 álayer_regularization_losses
âlayer_metrics
Ô	variables
Õtrainable_variables
Öregularization_losses
Ø__call__
+Ù&call_and_return_all_conditional_losses
'Ù"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block4_conv2_layer_call_fn_979075¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block4_conv2_layer_call_and_return_conditional_losses_979086¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ãnon_trainable_variables
älayers
åmetrics
 ælayer_regularization_losses
çlayer_metrics
Ú	variables
Ûtrainable_variables
Üregularization_losses
Þ__call__
+ß&call_and_return_all_conditional_losses
'ß"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block4_conv3_layer_call_fn_979095¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block4_conv3_layer_call_and_return_conditional_losses_979106¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ènon_trainable_variables
élayers
êmetrics
 ëlayer_regularization_losses
ìlayer_metrics
à	variables
átrainable_variables
âregularization_losses
ä__call__
+å&call_and_return_all_conditional_losses
'å"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block4_conv4_layer_call_fn_979115¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block4_conv4_layer_call_and_return_conditional_losses_979126¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ínon_trainable_variables
îlayers
ïmetrics
 ðlayer_regularization_losses
ñlayer_metrics
æ	variables
çtrainable_variables
èregularization_losses
ê__call__
+ë&call_and_return_all_conditional_losses
'ë"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_block4_pool_layer_call_fn_979131¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_block4_pool_layer_call_and_return_conditional_losses_979136¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ònon_trainable_variables
ólayers
ômetrics
 õlayer_regularization_losses
ölayer_metrics
ì	variables
ítrainable_variables
îregularization_losses
ð__call__
+ñ&call_and_return_all_conditional_losses
'ñ"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block5_conv1_layer_call_fn_979145¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block5_conv1_layer_call_and_return_conditional_losses_979156¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
÷non_trainable_variables
ølayers
ùmetrics
 úlayer_regularization_losses
ûlayer_metrics
ò	variables
ótrainable_variables
ôregularization_losses
ö__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block5_conv2_layer_call_fn_979165¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block5_conv2_layer_call_and_return_conditional_losses_979176¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ünon_trainable_variables
ýlayers
þmetrics
 ÿlayer_regularization_losses
layer_metrics
ø	variables
ùtrainable_variables
úregularization_losses
ü__call__
+ý&call_and_return_all_conditional_losses
'ý"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block5_conv3_layer_call_fn_979185¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block5_conv3_layer_call_and_return_conditional_losses_979196¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
þ	variables
ÿtrainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block5_conv4_layer_call_fn_979205¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block5_conv4_layer_call_and_return_conditional_losses_979216¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_block5_pool_layer_call_fn_979221¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_block5_pool_layer_call_and_return_conditional_losses_979226¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 

+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15
;16
<17
=18
>19
?20
@21
A22
B23
C24
D25
E26
F27
G28
H29
I30
J31"
trackable_list_wrapper
Æ
S0
T1
U2
V3
W4
X5
Y6
Z7
[8
\9
]10
^11
_12
`13
a14
b15
c16
d17
e18
f19
g20
h21"
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
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
 	variables"
_generic_user_object
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
-0
.1"
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
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
10
21"
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
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
90
:1"
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
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
A0
B1"
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
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
.
I0
J1"
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
&:$2Adam/dense_14/kernel/m
 :2Adam/dense_14/bias/m
':%	@2Adam/dense_13/kernel/m
 :@2Adam/dense_13/bias/m
&:$2Adam/dense_14/kernel/v
 :2Adam/dense_14/bias/v
':%	@2Adam/dense_13/kernel/v
 :@2Adam/dense_13/bias/vñ
!__inference__wrapped_model_974756Ë$+,-./0123456789:;<=>?@ABCDEFGHIJKL n¢k
d¢a
_\
,)
input_27ÿÿÿÿÿÿÿÿÿÈÈ
,)
input_28ÿÿÿÿÿÿÿÿÿÈÈ
ª "3ª0
.
dense_14"
dense_14ÿÿÿÿÿÿÿÿÿ¼
H__inference_block1_conv1_layer_call_and_return_conditional_losses_978876p+,9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿÈÈ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿÈÈ@
 
-__inference_block1_conv1_layer_call_fn_978865c+,9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿÈÈ
ª ""ÿÿÿÿÿÿÿÿÿÈÈ@¼
H__inference_block1_conv2_layer_call_and_return_conditional_losses_978896p-.9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿÈÈ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿÈÈ@
 
-__inference_block1_conv2_layer_call_fn_978885c-.9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿÈÈ@
ª ""ÿÿÿÿÿÿÿÿÿÈÈ@ê
G__inference_block1_pool_layer_call_and_return_conditional_losses_978906R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Â
,__inference_block1_pool_layer_call_fn_978901R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¹
H__inference_block2_conv1_layer_call_and_return_conditional_losses_978926m/07¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿdd@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿdd
 
-__inference_block2_conv1_layer_call_fn_978915`/07¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿdd@
ª "!ÿÿÿÿÿÿÿÿÿddº
H__inference_block2_conv2_layer_call_and_return_conditional_losses_978946n128¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿdd
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿdd
 
-__inference_block2_conv2_layer_call_fn_978935a128¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿdd
ª "!ÿÿÿÿÿÿÿÿÿddê
G__inference_block2_pool_layer_call_and_return_conditional_losses_978956R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Â
,__inference_block2_pool_layer_call_fn_978951R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
H__inference_block3_conv1_layer_call_and_return_conditional_losses_978976n348¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ22
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ22
 
-__inference_block3_conv1_layer_call_fn_978965a348¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ22
ª "!ÿÿÿÿÿÿÿÿÿ22º
H__inference_block3_conv2_layer_call_and_return_conditional_losses_978996n568¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ22
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ22
 
-__inference_block3_conv2_layer_call_fn_978985a568¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ22
ª "!ÿÿÿÿÿÿÿÿÿ22º
H__inference_block3_conv3_layer_call_and_return_conditional_losses_979016n788¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ22
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ22
 
-__inference_block3_conv3_layer_call_fn_979005a788¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ22
ª "!ÿÿÿÿÿÿÿÿÿ22º
H__inference_block3_conv4_layer_call_and_return_conditional_losses_979036n9:8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ22
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ22
 
-__inference_block3_conv4_layer_call_fn_979025a9:8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ22
ª "!ÿÿÿÿÿÿÿÿÿ22ê
G__inference_block3_pool_layer_call_and_return_conditional_losses_979046R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Â
,__inference_block3_pool_layer_call_fn_979041R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
H__inference_block4_conv1_layer_call_and_return_conditional_losses_979066n;<8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_block4_conv1_layer_call_fn_979055a;<8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿº
H__inference_block4_conv2_layer_call_and_return_conditional_losses_979086n=>8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_block4_conv2_layer_call_fn_979075a=>8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿº
H__inference_block4_conv3_layer_call_and_return_conditional_losses_979106n?@8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_block4_conv3_layer_call_fn_979095a?@8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿº
H__inference_block4_conv4_layer_call_and_return_conditional_losses_979126nAB8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_block4_conv4_layer_call_fn_979115aAB8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿê
G__inference_block4_pool_layer_call_and_return_conditional_losses_979136R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Â
,__inference_block4_pool_layer_call_fn_979131R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
H__inference_block5_conv1_layer_call_and_return_conditional_losses_979156nCD8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_block5_conv1_layer_call_fn_979145aCD8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿº
H__inference_block5_conv2_layer_call_and_return_conditional_losses_979176nEF8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_block5_conv2_layer_call_fn_979165aEF8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿº
H__inference_block5_conv3_layer_call_and_return_conditional_losses_979196nGH8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_block5_conv3_layer_call_fn_979185aGH8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿº
H__inference_block5_conv4_layer_call_and_return_conditional_losses_979216nIJ8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_block5_conv4_layer_call_fn_979205aIJ8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿê
G__inference_block5_pool_layer_call_and_return_conditional_losses_979226R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Â
,__inference_block5_pool_layer_call_fn_979221R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
D__inference_dense_13_layer_call_and_return_conditional_losses_978856]KL0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 }
)__inference_dense_13_layer_call_fn_978846PKL0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@¤
D__inference_dense_14_layer_call_and_return_conditional_losses_978446\ /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dense_14_layer_call_fn_978435O /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿß
V__inference_global_average_pooling2d_9_layer_call_and_return_conditional_losses_978837R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¶
;__inference_global_average_pooling2d_9_layer_call_fn_978831wR¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
D__inference_lambda_4_layer_call_and_return_conditional_losses_978412b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ@
"
inputs/1ÿÿÿÿÿÿÿÿÿ@

 
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ô
D__inference_lambda_4_layer_call_and_return_conditional_losses_978426b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ@
"
inputs/1ÿÿÿÿÿÿÿÿÿ@

 
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 «
)__inference_lambda_4_layer_call_fn_978392~b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ@
"
inputs/1ÿÿÿÿÿÿÿÿÿ@

 
p 
ª "ÿÿÿÿÿÿÿÿÿ«
)__inference_lambda_4_layer_call_fn_978398~b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ@
"
inputs/1ÿÿÿÿÿÿÿÿÿ@

 
p
ª "ÿÿÿÿÿÿÿÿÿÙ
D__inference_model_13_layer_call_and_return_conditional_losses_976355"+,-./0123456789:;<=>?@ABCDEFGHIJKLC¢@
9¢6
,)
input_29ÿÿÿÿÿÿÿÿÿÈÈ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 Ù
D__inference_model_13_layer_call_and_return_conditional_losses_976430"+,-./0123456789:;<=>?@ABCDEFGHIJKLC¢@
9¢6
,)
input_29ÿÿÿÿÿÿÿÿÿÈÈ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ×
D__inference_model_13_layer_call_and_return_conditional_losses_978257"+,-./0123456789:;<=>?@ABCDEFGHIJKLA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿÈÈ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ×
D__inference_model_13_layer_call_and_return_conditional_losses_978386"+,-./0123456789:;<=>?@ABCDEFGHIJKLA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿÈÈ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ±
)__inference_model_13_layer_call_fn_975976"+,-./0123456789:;<=>?@ABCDEFGHIJKLC¢@
9¢6
,)
input_29ÿÿÿÿÿÿÿÿÿÈÈ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ@±
)__inference_model_13_layer_call_fn_976280"+,-./0123456789:;<=>?@ABCDEFGHIJKLC¢@
9¢6
,)
input_29ÿÿÿÿÿÿÿÿÿÈÈ
p

 
ª "ÿÿÿÿÿÿÿÿÿ@¯
)__inference_model_13_layer_call_fn_978055"+,-./0123456789:;<=>?@ABCDEFGHIJKLA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿÈÈ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ@¯
)__inference_model_13_layer_call_fn_978128"+,-./0123456789:;<=>?@ABCDEFGHIJKLA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿÈÈ
p

 
ª "ÿÿÿÿÿÿÿÿÿ@
D__inference_model_14_layer_call_and_return_conditional_losses_977153Å$+,-./0123456789:;<=>?@ABCDEFGHIJKL v¢s
l¢i
_\
,)
input_27ÿÿÿÿÿÿÿÿÿÈÈ
,)
input_28ÿÿÿÿÿÿÿÿÿÈÈ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
D__inference_model_14_layer_call_and_return_conditional_losses_977268Å$+,-./0123456789:;<=>?@ABCDEFGHIJKL v¢s
l¢i
_\
,)
input_27ÿÿÿÿÿÿÿÿÿÈÈ
,)
input_28ÿÿÿÿÿÿÿÿÿÈÈ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
D__inference_model_14_layer_call_and_return_conditional_losses_977665Å$+,-./0123456789:;<=>?@ABCDEFGHIJKL v¢s
l¢i
_\
,)
inputs/0ÿÿÿÿÿÿÿÿÿÈÈ
,)
inputs/1ÿÿÿÿÿÿÿÿÿÈÈ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
D__inference_model_14_layer_call_and_return_conditional_losses_977902Å$+,-./0123456789:;<=>?@ABCDEFGHIJKL v¢s
l¢i
_\
,)
inputs/0ÿÿÿÿÿÿÿÿÿÈÈ
,)
inputs/1ÿÿÿÿÿÿÿÿÿÈÈ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 æ
)__inference_model_14_layer_call_fn_976652¸$+,-./0123456789:;<=>?@ABCDEFGHIJKL v¢s
l¢i
_\
,)
input_27ÿÿÿÿÿÿÿÿÿÈÈ
,)
input_28ÿÿÿÿÿÿÿÿÿÈÈ
p 

 
ª "ÿÿÿÿÿÿÿÿÿæ
)__inference_model_14_layer_call_fn_977038¸$+,-./0123456789:;<=>?@ABCDEFGHIJKL v¢s
l¢i
_\
,)
input_27ÿÿÿÿÿÿÿÿÿÈÈ
,)
input_28ÿÿÿÿÿÿÿÿÿÈÈ
p

 
ª "ÿÿÿÿÿÿÿÿÿæ
)__inference_model_14_layer_call_fn_977350¸$+,-./0123456789:;<=>?@ABCDEFGHIJKL v¢s
l¢i
_\
,)
inputs/0ÿÿÿÿÿÿÿÿÿÈÈ
,)
inputs/1ÿÿÿÿÿÿÿÿÿÈÈ
p 

 
ª "ÿÿÿÿÿÿÿÿÿæ
)__inference_model_14_layer_call_fn_977428¸$+,-./0123456789:;<=>?@ABCDEFGHIJKL v¢s
l¢i
_\
,)
inputs/0ÿÿÿÿÿÿÿÿÿÈÈ
,)
inputs/1ÿÿÿÿÿÿÿÿÿÈÈ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
$__inference_signature_wrapper_977982ß$+,-./0123456789:;<=>?@ABCDEFGHIJKL ¢~
¢ 
wªt
8
input_27,)
input_27ÿÿÿÿÿÿÿÿÿÈÈ
8
input_28,)
input_28ÿÿÿÿÿÿÿÿÿÈÈ"3ª0
.
dense_14"
dense_14ÿÿÿÿÿÿÿÿÿÝ
A__inference_vgg19_layer_call_and_return_conditional_losses_975713 +,-./0123456789:;<=>?@ABCDEFGHIJC¢@
9¢6
,)
input_30ÿÿÿÿÿÿÿÿÿÈÈ
p 

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ý
A__inference_vgg19_layer_call_and_return_conditional_losses_975802 +,-./0123456789:;<=>?@ABCDEFGHIJC¢@
9¢6
,)
input_30ÿÿÿÿÿÿÿÿÿÈÈ
p

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Û
A__inference_vgg19_layer_call_and_return_conditional_losses_978705 +,-./0123456789:;<=>?@ABCDEFGHIJA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿÈÈ
p 

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Û
A__inference_vgg19_layer_call_and_return_conditional_losses_978826 +,-./0123456789:;<=>?@ABCDEFGHIJA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿÈÈ
p

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 µ
&__inference_vgg19_layer_call_fn_975168 +,-./0123456789:;<=>?@ABCDEFGHIJC¢@
9¢6
,)
input_30ÿÿÿÿÿÿÿÿÿÈÈ
p 

 
ª "!ÿÿÿÿÿÿÿÿÿµ
&__inference_vgg19_layer_call_fn_975624 +,-./0123456789:;<=>?@ABCDEFGHIJC¢@
9¢6
,)
input_30ÿÿÿÿÿÿÿÿÿÈÈ
p

 
ª "!ÿÿÿÿÿÿÿÿÿ³
&__inference_vgg19_layer_call_fn_978515 +,-./0123456789:;<=>?@ABCDEFGHIJA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿÈÈ
p 

 
ª "!ÿÿÿÿÿÿÿÿÿ³
&__inference_vgg19_layer_call_fn_978584 +,-./0123456789:;<=>?@ABCDEFGHIJA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿÈÈ
p

 
ª "!ÿÿÿÿÿÿÿÿÿ
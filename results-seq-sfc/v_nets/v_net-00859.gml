graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 859
  arrival_time 17721.07099935226
  lifetime 229.56351913126315
  num_nodes 12
  type "path"
  node [
    id 0
    label "0"
    cpu 7
    gpu 21
    rom 11
  ]
  node [
    id 1
    label "1"
    cpu 39
    gpu 9
    rom 30
  ]
  node [
    id 2
    label "2"
    cpu 41
    gpu 8
    rom 48
  ]
  node [
    id 3
    label "3"
    cpu 25
    gpu 16
    rom 19
  ]
  node [
    id 4
    label "4"
    cpu 22
    gpu 45
    rom 19
  ]
  node [
    id 5
    label "5"
    cpu 15
    gpu 39
    rom 46
  ]
  node [
    id 6
    label "6"
    cpu 34
    gpu 28
    rom 8
  ]
  node [
    id 7
    label "7"
    cpu 37
    gpu 46
    rom 10
  ]
  node [
    id 8
    label "8"
    cpu 23
    gpu 40
    rom 34
  ]
  node [
    id 9
    label "9"
    cpu 37
    gpu 39
    rom 1
  ]
  node [
    id 10
    label "10"
    cpu 25
    gpu 7
    rom 22
  ]
  node [
    id 11
    label "11"
    cpu 39
    gpu 22
    rom 13
  ]
  edge [
    source 0
    target 1
    bw 2
  ]
  edge [
    source 1
    target 2
    bw 47
  ]
  edge [
    source 2
    target 3
    bw 32
  ]
  edge [
    source 3
    target 4
    bw 45
  ]
  edge [
    source 4
    target 5
    bw 19
  ]
  edge [
    source 5
    target 6
    bw 19
  ]
  edge [
    source 6
    target 7
    bw 4
  ]
  edge [
    source 7
    target 8
    bw 28
  ]
  edge [
    source 8
    target 9
    bw 48
  ]
  edge [
    source 9
    target 10
    bw 37
  ]
  edge [
    source 10
    target 11
    bw 16
  ]
]

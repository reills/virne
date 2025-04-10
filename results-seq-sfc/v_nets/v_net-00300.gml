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
  id 300
  arrival_time 5772.4473383339255
  lifetime 409.11291070029506
  num_nodes 9
  type "path"
  node [
    id 0
    label "0"
    cpu 14
    gpu 21
    rom 24
  ]
  node [
    id 1
    label "1"
    cpu 42
    gpu 32
    rom 12
  ]
  node [
    id 2
    label "2"
    cpu 48
    gpu 46
    rom 48
  ]
  node [
    id 3
    label "3"
    cpu 16
    gpu 18
    rom 26
  ]
  node [
    id 4
    label "4"
    cpu 38
    gpu 0
    rom 4
  ]
  node [
    id 5
    label "5"
    cpu 41
    gpu 45
    rom 8
  ]
  node [
    id 6
    label "6"
    cpu 29
    gpu 22
    rom 27
  ]
  node [
    id 7
    label "7"
    cpu 41
    gpu 0
    rom 46
  ]
  node [
    id 8
    label "8"
    cpu 0
    gpu 18
    rom 5
  ]
  edge [
    source 0
    target 1
    bw 32
  ]
  edge [
    source 1
    target 2
    bw 2
  ]
  edge [
    source 2
    target 3
    bw 35
  ]
  edge [
    source 3
    target 4
    bw 42
  ]
  edge [
    source 4
    target 5
    bw 12
  ]
  edge [
    source 5
    target 6
    bw 22
  ]
  edge [
    source 6
    target 7
    bw 15
  ]
  edge [
    source 7
    target 8
    bw 44
  ]
]

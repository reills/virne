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
  id 1184
  arrival_time 24419.144888689247
  lifetime 1125.3360464950701
  num_nodes 13
  type "path"
  node [
    id 0
    label "0"
    cpu 42
    gpu 45
    rom 9
  ]
  node [
    id 1
    label "1"
    cpu 2
    gpu 21
    rom 13
  ]
  node [
    id 2
    label "2"
    cpu 48
    gpu 15
    rom 47
  ]
  node [
    id 3
    label "3"
    cpu 46
    gpu 36
    rom 33
  ]
  node [
    id 4
    label "4"
    cpu 22
    gpu 28
    rom 37
  ]
  node [
    id 5
    label "5"
    cpu 14
    gpu 48
    rom 40
  ]
  node [
    id 6
    label "6"
    cpu 33
    gpu 14
    rom 35
  ]
  node [
    id 7
    label "7"
    cpu 18
    gpu 21
    rom 3
  ]
  node [
    id 8
    label "8"
    cpu 23
    gpu 49
    rom 43
  ]
  node [
    id 9
    label "9"
    cpu 44
    gpu 26
    rom 25
  ]
  node [
    id 10
    label "10"
    cpu 29
    gpu 5
    rom 45
  ]
  node [
    id 11
    label "11"
    cpu 36
    gpu 20
    rom 13
  ]
  node [
    id 12
    label "12"
    cpu 37
    gpu 17
    rom 28
  ]
  edge [
    source 0
    target 1
    bw 20
  ]
  edge [
    source 1
    target 2
    bw 14
  ]
  edge [
    source 2
    target 3
    bw 1
  ]
  edge [
    source 3
    target 4
    bw 25
  ]
  edge [
    source 4
    target 5
    bw 15
  ]
  edge [
    source 5
    target 6
    bw 46
  ]
  edge [
    source 6
    target 7
    bw 17
  ]
  edge [
    source 7
    target 8
    bw 9
  ]
  edge [
    source 8
    target 9
    bw 7
  ]
  edge [
    source 9
    target 10
    bw 14
  ]
  edge [
    source 10
    target 11
    bw 14
  ]
  edge [
    source 11
    target 12
    bw 4
  ]
]

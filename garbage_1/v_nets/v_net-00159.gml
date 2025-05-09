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
  id 159
  arrival_time 3043.107857513736
  lifetime 1402.7292122891151
  num_nodes 12
  type "path"
  node [
    id 0
    label "0"
    cpu 14
    gpu 30
    rom 12
  ]
  node [
    id 1
    label "1"
    cpu 15
    gpu 49
    rom 7
  ]
  node [
    id 2
    label "2"
    cpu 6
    gpu 36
    rom 7
  ]
  node [
    id 3
    label "3"
    cpu 6
    gpu 19
    rom 28
  ]
  node [
    id 4
    label "4"
    cpu 12
    gpu 8
    rom 21
  ]
  node [
    id 5
    label "5"
    cpu 28
    gpu 11
    rom 38
  ]
  node [
    id 6
    label "6"
    cpu 5
    gpu 19
    rom 44
  ]
  node [
    id 7
    label "7"
    cpu 15
    gpu 18
    rom 41
  ]
  node [
    id 8
    label "8"
    cpu 21
    gpu 30
    rom 30
  ]
  node [
    id 9
    label "9"
    cpu 34
    gpu 25
    rom 30
  ]
  node [
    id 10
    label "10"
    cpu 3
    gpu 28
    rom 32
  ]
  node [
    id 11
    label "11"
    cpu 40
    gpu 44
    rom 19
  ]
  edge [
    source 0
    target 1
    bw 35
  ]
  edge [
    source 1
    target 2
    bw 31
  ]
  edge [
    source 2
    target 3
    bw 7
  ]
  edge [
    source 3
    target 4
    bw 50
  ]
  edge [
    source 4
    target 5
    bw 42
  ]
  edge [
    source 5
    target 6
    bw 22
  ]
  edge [
    source 6
    target 7
    bw 35
  ]
  edge [
    source 7
    target 8
    bw 40
  ]
  edge [
    source 8
    target 9
    bw 37
  ]
  edge [
    source 9
    target 10
    bw 1
  ]
  edge [
    source 10
    target 11
    bw 43
  ]
]

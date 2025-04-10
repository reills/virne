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
  id 219
  arrival_time 4045.0929667501673
  lifetime 2174.014386985031
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 0
    gpu 43
    rom 49
  ]
  node [
    id 1
    label "1"
    cpu 33
    gpu 28
    rom 31
  ]
  node [
    id 2
    label "2"
    cpu 49
    gpu 44
    rom 38
  ]
  node [
    id 3
    label "3"
    cpu 27
    gpu 40
    rom 44
  ]
  node [
    id 4
    label "4"
    cpu 23
    gpu 3
    rom 18
  ]
  node [
    id 5
    label "5"
    cpu 12
    gpu 0
    rom 9
  ]
  node [
    id 6
    label "6"
    cpu 36
    gpu 21
    rom 33
  ]
  node [
    id 7
    label "7"
    cpu 44
    gpu 20
    rom 16
  ]
  node [
    id 8
    label "8"
    cpu 41
    gpu 24
    rom 7
  ]
  node [
    id 9
    label "9"
    cpu 26
    gpu 13
    rom 6
  ]
  edge [
    source 0
    target 1
    bw 6
  ]
  edge [
    source 1
    target 2
    bw 22
  ]
  edge [
    source 2
    target 3
    bw 19
  ]
  edge [
    source 3
    target 4
    bw 49
  ]
  edge [
    source 4
    target 5
    bw 14
  ]
  edge [
    source 5
    target 6
    bw 21
  ]
  edge [
    source 6
    target 7
    bw 21
  ]
  edge [
    source 7
    target 8
    bw 40
  ]
  edge [
    source 8
    target 9
    bw 35
  ]
]

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
  id 621
  arrival_time 12815.107016012174
  lifetime 927.1906840266635
  num_nodes 11
  type "path"
  node [
    id 0
    label "0"
    cpu 43
    gpu 1
    rom 28
  ]
  node [
    id 1
    label "1"
    cpu 28
    gpu 14
    rom 31
  ]
  node [
    id 2
    label "2"
    cpu 5
    gpu 34
    rom 25
  ]
  node [
    id 3
    label "3"
    cpu 38
    gpu 49
    rom 29
  ]
  node [
    id 4
    label "4"
    cpu 44
    gpu 25
    rom 27
  ]
  node [
    id 5
    label "5"
    cpu 0
    gpu 38
    rom 6
  ]
  node [
    id 6
    label "6"
    cpu 13
    gpu 37
    rom 46
  ]
  node [
    id 7
    label "7"
    cpu 30
    gpu 22
    rom 4
  ]
  node [
    id 8
    label "8"
    cpu 28
    gpu 1
    rom 43
  ]
  node [
    id 9
    label "9"
    cpu 23
    gpu 46
    rom 9
  ]
  node [
    id 10
    label "10"
    cpu 24
    gpu 26
    rom 3
  ]
  edge [
    source 0
    target 1
    bw 17
  ]
  edge [
    source 1
    target 2
    bw 25
  ]
  edge [
    source 2
    target 3
    bw 23
  ]
  edge [
    source 3
    target 4
    bw 21
  ]
  edge [
    source 4
    target 5
    bw 35
  ]
  edge [
    source 5
    target 6
    bw 35
  ]
  edge [
    source 6
    target 7
    bw 11
  ]
  edge [
    source 7
    target 8
    bw 23
  ]
  edge [
    source 8
    target 9
    bw 39
  ]
  edge [
    source 9
    target 10
    bw 26
  ]
]

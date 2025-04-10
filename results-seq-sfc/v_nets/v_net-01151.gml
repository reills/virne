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
  id 1151
  arrival_time 23973.438905261108
  lifetime 602.7873332406665
  num_nodes 13
  type "path"
  node [
    id 0
    label "0"
    cpu 39
    gpu 39
    rom 32
  ]
  node [
    id 1
    label "1"
    cpu 37
    gpu 12
    rom 14
  ]
  node [
    id 2
    label "2"
    cpu 4
    gpu 36
    rom 31
  ]
  node [
    id 3
    label "3"
    cpu 27
    gpu 28
    rom 44
  ]
  node [
    id 4
    label "4"
    cpu 46
    gpu 32
    rom 15
  ]
  node [
    id 5
    label "5"
    cpu 50
    gpu 28
    rom 37
  ]
  node [
    id 6
    label "6"
    cpu 24
    gpu 16
    rom 10
  ]
  node [
    id 7
    label "7"
    cpu 31
    gpu 39
    rom 16
  ]
  node [
    id 8
    label "8"
    cpu 21
    gpu 34
    rom 34
  ]
  node [
    id 9
    label "9"
    cpu 16
    gpu 48
    rom 44
  ]
  node [
    id 10
    label "10"
    cpu 37
    gpu 44
    rom 20
  ]
  node [
    id 11
    label "11"
    cpu 1
    gpu 37
    rom 27
  ]
  node [
    id 12
    label "12"
    cpu 14
    gpu 2
    rom 14
  ]
  edge [
    source 0
    target 1
    bw 12
  ]
  edge [
    source 1
    target 2
    bw 42
  ]
  edge [
    source 2
    target 3
    bw 38
  ]
  edge [
    source 3
    target 4
    bw 28
  ]
  edge [
    source 4
    target 5
    bw 33
  ]
  edge [
    source 5
    target 6
    bw 37
  ]
  edge [
    source 6
    target 7
    bw 43
  ]
  edge [
    source 7
    target 8
    bw 19
  ]
  edge [
    source 8
    target 9
    bw 43
  ]
  edge [
    source 9
    target 10
    bw 36
  ]
  edge [
    source 10
    target 11
    bw 12
  ]
  edge [
    source 11
    target 12
    bw 25
  ]
]
